//!
//! A lock-free thread-owned queue whereby tasks are taken by stealers in entirety via buffer swapping. This is meant to be used [`thread_local`] paired with [`tokio::task::spawn`] as a take-all batching mechanism that outperforms [`crossbeam::deque::Worker`], and [`tokio::sync::mpsc`] for batching.
//!
//! ## Example
//!
//! ```
//! use swap_queue::Worker;
//! use tokio::{
//!   runtime::Handle,
//!   sync::oneshot::{channel, Sender},
//! };
//!
//! // Jemalloc makes this library substantially faster
//! #[global_allocator]
//! static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
//!
//! // Worker needs to be thread local because it is !Sync
//! thread_local! {
//!   static QUEUE: Worker<(u64, Sender<u64>)> = Worker::new();
//! }
//!
//! // This mechanism will batch optimally without overhead within an async-context because take will poll after everything else scheduled
//! async fn push_echo(i: u64) -> u64 {
//!   {
//!     let (tx, rx) = channel();
//!
//!     QUEUE.with(|queue| {
//!       // A new stealer is issued for every buffer swap
//!       if let Some(stealer) = queue.push((i, tx)) {
//!         Handle::current().spawn(async move {
//!           // Take the underlying buffer in entirety.
//!           let batch = stealer.take().await;
//!
//!           // Some sort of batched operation, such as a database load
//!
//!           batch.into_iter().for_each(|(i, tx)| {
//!             tx.send(i).ok();
//!           });
//!         });
//!       }
//!     });
//!
//!     rx
//!   }
//!   .await
//!   .unwrap()
//! }
//! ```

use crossbeam::{
  epoch::{self, Atomic, Owned, Shared},
  utils::CachePadded,
};
use futures::executor::block_on;
use std::{cell::Cell, fmt, marker::PhantomData, mem, ptr, sync::Arc};
use tokio::sync::oneshot::{channel, Receiver, Sender};

#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(loom))]
use std::sync::atomic::{AtomicUsize, Ordering};

// Current buffer index
const BUFFER_IDX: usize = 1 << 0;

// Designates that write is in progress
const WRITE_IN_PROGRESS: usize = 1 << 1;

// Designates how many bits are set aside for flags
const FLAGS_SHIFT: usize = 1;

// Slot increments both for reads and writes, therefore we shift slot an extra bit to extract length
const LENGTH_SHIFT: usize = FLAGS_SHIFT + 1;

// Minimum buffer capacity.
const MIN_CAP: usize = 64;

/// A buffer that holds tasks in a worker queue.
///
/// This is just a pointer to the buffer and its length - dropping an instance of this struct will
/// *not* deallocate the buffer.
struct Buffer<T> {
  /// Slot that represents the index offset and buffer idx
  slot: usize,

  /// Pointer to the allocated memory.
  ptr: *mut T,

  /// Capacity of the buffer. Always a power of two.
  cap: usize,
}

unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Send> Sync for Buffer<T> {}

impl<T> Buffer<T> {
  /// Allocates a new buffer with the specified capacity.
  fn alloc(slot: usize, cap: usize) -> Buffer<T> {
    debug_assert_eq!(cap, cap.next_power_of_two());

    let mut v = Vec::with_capacity(cap);
    let ptr = v.as_mut_ptr();
    mem::forget(v);

    Buffer { slot, ptr, cap }
  }

  /// Deallocates the buffer.
  unsafe fn dealloc(self) {
    drop(Vec::from_raw_parts(self.ptr, 0, self.cap));
  }

  /// Returns a pointer to the task at the specified `index`.
  unsafe fn at(&self, index: usize) -> *mut T {
    // `self.cap` is always a power of two.
    self.ptr.offset((index & (self.cap - 1)) as isize)
  }

  /// Writes `task` into the specified `index`.
  unsafe fn write(&self, index: usize, task: T) {
    ptr::write_volatile(self.at(index), task)
  }

  unsafe fn to_vec(self, length: usize) -> Vec<T> {
    let Buffer { ptr, cap, .. } = self;
    Vec::from_raw_parts(ptr, length, cap)
  }
}

impl<T> Clone for Buffer<T> {
  fn clone(&self) -> Buffer<T> {
    Buffer {
      slot: self.slot,
      ptr: self.ptr,
      cap: self.cap,
    }
  }
}

impl<T> Copy for Buffer<T> {}

fn slot_delta(a: usize, b: usize) -> usize {
  if a < b {
    ((usize::MAX - b) >> LENGTH_SHIFT) + (a >> LENGTH_SHIFT)
  } else {
    (a >> LENGTH_SHIFT) - (b >> LENGTH_SHIFT)
  }
}

struct Inner<T> {
  slot: AtomicUsize,
  buffers: (
    CachePadded<Atomic<Buffer<T>>>,
    CachePadded<Atomic<Buffer<T>>>,
  ),
}

impl<T> Inner<T> {
  fn get_buffer(&self, slot: usize) -> &CachePadded<Atomic<Buffer<T>>> {
    if slot & BUFFER_IDX == 0 {
      &self.buffers.0
    } else {
      &self.buffers.1
    }
  }
}

impl<T> Drop for Inner<T> {
  fn drop(&mut self) {
    let slot = self.slot.load(Ordering::Acquire);

    let guard = &epoch::pin();
    let buffer = self
      .get_buffer(slot)
      .swap(Shared::null(), Ordering::Acquire, guard);

    if !buffer.is_null() {
      unsafe {
        guard.defer_unchecked(move || {
          let buffer = *buffer.into_owned();
          let length = slot_delta(slot, buffer.slot);

          // Go through the buffer from front to back and drop all tasks in the queue.
          for i in 0..length {
            buffer.at(i).drop_in_place();
          }

          // Free the memory allocated by the buffer.
          buffer.dealloc();
        });
      }
    }
  }
}

/// A thread-owned worker queue that writes to a swappable buffer using atomic slotting
///
/// # Examples
///
/// ```
/// use swap_queue::Worker;
///
/// let w = Worker::new();
/// let s = w.push(1).unwrap();
/// w.push(2);
/// w.push(3);
/// assert_eq!(s.take_blocking(), vec![1, 2, 3]);
///
/// let s = w.push(4).unwrap();
/// w.push(5);
/// w.push(6);
/// assert_eq!(s.take_blocking(), vec![4, 5, 6]);
/// ```

pub struct Worker<T> {
  /// A reference to the inner representation of the queue.
  inner: Arc<CachePadded<Inner<T>>>,
  /// A copy of `inner.buffer` for quick access.
  buffer: Cell<Buffer<T>>,
  /// Send handle corresponding to the current Stealer
  tx: Cell<Option<Sender<Vec<T>>>>,
  /// Indicates that the worker cannot be shared among threads.
  _marker: PhantomData<*mut ()>,
}

unsafe impl<T: Send> Send for Worker<T> {}

impl<T> Worker<T> {
  /// Creates a new Worker queue.
  ///
  /// Tasks are pushed and the underlying buffer is swapped and converted back into a `Vec<T>` when taken by a stealer.
  ///
  /// # Examples
  ///
  /// ```
  /// use swap_queue::Worker;
  ///
  /// let w = Worker::<i32>::new();
  /// ```
  pub fn new() -> Worker<T> {
    let buffer = Buffer::alloc(0, MIN_CAP);

    let inner = Arc::new(CachePadded::new(Inner {
      slot: AtomicUsize::new(0),
      buffers: (
        CachePadded::new(Atomic::new(buffer)),
        CachePadded::new(Atomic::null()),
      ),
    }));

    Worker {
      inner,
      buffer: Cell::new(buffer),
      tx: Cell::new(None),
      _marker: PhantomData,
    }
  }

  /// Resizes the internal buffer to the new capacity of `new_cap`.
  unsafe fn resize(&self, buffer: &mut Buffer<T>, slot: usize) {
    let length = slot_delta(slot, buffer.slot);

    // Allocate a new buffer and copy data from the old buffer to the new one.
    let new = Buffer::alloc(buffer.slot, buffer.cap * 2);

    ptr::copy_nonoverlapping(buffer.at(0), new.at(0), length);

    self.buffer.set(new);

    let old = std::mem::replace(buffer, new);

    self
      .inner
      .get_buffer(slot)
      .store(Owned::new(new), Ordering::Release);

    old.dealloc();
  }

  /// Push a task to the queue and returns a Stealer if empty
  pub fn push(&self, task: T) -> Option<Stealer<T>> {
    let slot = self
      .inner
      .slot
      .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

    let idx = slot & BUFFER_IDX;

    let mut buffer = self.buffer.get();

    // Is buffer still current?
    let index = if idx == buffer.slot & BUFFER_IDX {
      let index = slot_delta(slot, buffer.slot);

      // Is the queue full?
      if index >= buffer.cap {
        // Yes. Grow the underlying buffer.
        unsafe {
          self.resize(&mut buffer, slot);
        }
      }

      index
    } else {
      buffer = Buffer::alloc(slot, buffer.cap);

      self
        .inner
        .get_buffer(slot)
        .store(Owned::new(buffer), Ordering::Release);

      self.buffer.set(buffer);

      0
    };

    unsafe {
      buffer.write(index, task);
    }

    let slot = self
      .inner
      .slot
      .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

    // If the buffer idx was swapped by the stealer while a write is in progress, then it is waiting on receive
    if idx != slot & BUFFER_IDX {
      let (tx, rx) = channel();
      let tx = self.tx.replace(Some(tx)).unwrap();

      // Send buffer as vec to receiver
      tx.send(unsafe { buffer.to_vec(index) }).ok();

      Some(Stealer {
        rx,
        inner: self.inner.clone(),
      })
    } else if index == 0 {
      let (tx, rx) = channel();
      self.tx.set(Some(tx));

      Some(Stealer {
        rx,
        inner: self.inner.clone(),
      })
    } else {
      None
    }
  }
}

impl<T> Default for Worker<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T> fmt::Debug for Worker<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.pad("Worker { .. }")
  }
}

/// For every buffer swapped in a stealer is issued that can later swap out and take ownership of that buffer
pub struct Stealer<T> {
  /// Buffer receiver to be used when waiting on writes
  rx: Receiver<Vec<T>>,
  /// A reference to the inner representation of the queue.
  inner: Arc<CachePadded<Inner<T>>>,
}

unsafe impl<T: Send> Send for Stealer<T> {}
unsafe impl<T: Send> Sync for Stealer<T> {}

impl<T> Stealer<T> {
  /// Take the entire queue via swapping the underlying buffer and converting into a `Vec<T>`
  pub async fn take(self) -> Vec<T> {
    let Stealer { rx, inner } = self;

    let slot = inner.slot.fetch_xor(BUFFER_IDX, Ordering::Relaxed);

    // Worker will see the buffer has swapped when confirming length increment
    if slot & WRITE_IN_PROGRESS == WRITE_IN_PROGRESS {
      // Writer can never be dropped mid-write, therefore RecvError cannot occur
      rx.await.unwrap()
    } else {
      let guard = &epoch::pin();

      let buffer = inner
        .get_buffer(slot)
        .swap(Shared::null(), Ordering::Acquire, guard);

      unsafe {
        let buffer = *buffer.into_owned();
        buffer.to_vec(slot_delta(slot, buffer.slot))
      }
    }
  }

  /// Take the entire queue via swapping the underlying buffer and converting into a `Vec<T>`, blocking only if write in progress. This is always non-blocking if never moved to a new thread.
  pub fn take_blocking(self) -> Vec<T> {
    let Stealer { rx, inner } = self;

    let slot = inner.slot.fetch_xor(BUFFER_IDX, Ordering::Relaxed);

    // Worker will see the buffer has swapped when confirming length increment
    // It's not possible for this to be write in progress when called from the same thread as the queue
    if slot & WRITE_IN_PROGRESS == WRITE_IN_PROGRESS {
      // Writer can never be dropped mid-write, therefore RecvError cannot occur
      block_on(rx).unwrap()
    } else {
      let guard = &epoch::pin();

      let buffer = inner
        .get_buffer(slot)
        .swap(Shared::null(), Ordering::Acquire, guard);

      unsafe {
        let buffer = *buffer.into_owned();
        buffer.to_vec(slot_delta(slot, buffer.slot))
      }
    }
  }
}

impl<T> fmt::Debug for Stealer<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.pad("Stealer { .. }")
  }
}

#[cfg(all(test))]
mod tests {
  use super::*;

  #[cfg(loom)]
  use loom::thread;

  #[cfg(not(loom))]
  use std::thread;

  macro_rules! model {
    ($test:block) => {
      #[cfg(loom)]
      loom::model(|| $test);

      #[cfg(not(loom))]
      $test
    };
  }

  #[test]
  fn slot_wraps_around() {
    let delta = slot_delta(1 << LENGTH_SHIFT, usize::MAX);

    assert_eq!(delta, 1);
  }

  #[test]
  fn it_resizes() {
    model!({
      let queue = Worker::new();
      let stealer = queue.push(0).unwrap();

      for i in 1..128 {
        queue.push(i);
      }

      let batch = stealer.take_blocking();
      let expected = (0..128).collect::<Vec<i32>>();

      assert_eq!(batch, expected);
    });
  }

  #[cfg(not(loom))]
  #[tokio::test]
  async fn stealer_takes() {
    let queue = Worker::new();
    let stealer = queue.push(0).unwrap();

    for i in 1..1024 {
      queue.push(i);
    }

    let batch = stealer.take().await;
    let expected = (0..1024).collect::<Vec<i32>>();

    assert_eq!(batch, expected);
  }

  #[test]
  fn stealer_takes_blocking() {
    model!({
      let queue = Worker::new();
      let stealer = queue.push(0).unwrap();

      for i in 1..128 {
        queue.push(i);
      }

      thread::spawn(move || {
        stealer.take_blocking();
      })
      .join()
      .unwrap();
    });
  }

  #[cfg(not(loom))]
  #[tokio::test]
  async fn worker_drops() {
    let queue = Worker::new();
    let stealer = queue.push(0).unwrap();

    for i in 1..128 {
      queue.push(i);
    }

    drop(queue);

    let batch = stealer.take().await;
    let expected = (0..128).collect::<Vec<i32>>();

    assert_eq!(batch, expected);
  }

  #[cfg(loom)]
  #[tokio::test]
  async fn worker_drops() {
    loom::model(|| {
      let queue = Worker::new();
      let stealer = queue.push(0).unwrap();

      for i in 1..128 {
        queue.push(i);
      }

      drop(queue);

      let batch = stealer.take_blocking();
      let expected = (0..128).collect::<Vec<i32>>();

      assert_eq!(batch, expected);
    });
  }
}
