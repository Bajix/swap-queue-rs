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
//! // This mechanism will batch optimally without overhead within an async-context because spawn will happen after things already scheduled
//! async fn push_echo(i: u64) -> u64 {
//!   {
//!     let (tx, rx) = channel();
//!
//!     QUEUE.with(|queue| {
//!       // A new stealer is returned whenever the buffer is new or was empty
//!       if let Some(stealer) = queue.push((i, tx)) {
//!         Handle::current().spawn(async move {
//!           // Take the underlying buffer in entirety; the next push will return a new Stealer
//!           let batch = stealer.take().await;
//!
//!           // Some sort of batched operation, such as a database query
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
  epoch::{self, Atomic, Owned},
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
/// // this is non-blocking because it's called on the same thread as Worker; a write in progress is not possible
/// assert_eq!(s.take_blocking(), vec![1, 2, 3]);
///
/// let s = w.push(4).unwrap();
/// w.push(5);
/// w.push(6);
/// // this is identical to [`Stealer::take_blocking`]
/// let batch: Vec<_> = s.into();
/// assert_eq!(batch, vec![4, 5, 6]);
/// ```

enum Flavor {
  Unbounded,
  AutoBatched { batch_size: usize },
}

pub struct Worker<T> {
  flavor: Flavor,
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
      flavor: Flavor::Unbounded,
      inner,
      buffer: Cell::new(buffer),
      tx: Cell::new(None),
      _marker: PhantomData,
    }
  }

  /// Creates an auto-batched Worker queue with fixed-length buffers. At capacity, the buffer is swapped out and ownership taken by the returned Stealer. Batch size must be a power of 2
  ///
  /// # Examples
  ///
  /// ```
  /// use swap_queue::Worker;
  ///
  /// let w = Worker::<i32>::auto_batched(64);
  /// ```
  pub fn auto_batched(batch_size: usize) -> Worker<T> {
    debug_assert!(batch_size.ge(&64), "batch_size must be at least 64");
    debug_assert_eq!(
      batch_size,
      batch_size.next_power_of_two(),
      "batch_size must be a power of 2"
    );

    let buffer = Buffer::alloc(0, MIN_CAP);

    let inner = Arc::new(CachePadded::new(Inner {
      slot: AtomicUsize::new(0),
      buffers: (
        CachePadded::new(Atomic::new(buffer)),
        CachePadded::new(Atomic::null()),
      ),
    }));

    Worker {
      flavor: Flavor::AutoBatched { batch_size },
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

  fn replace_buffer(&self, buffer: &mut Buffer<T>, slot: usize, cap: usize) -> Buffer<T> {
    let new = Buffer::alloc(slot.to_owned(), cap);

    self
      .inner
      .get_buffer(slot)
      .store(Owned::new(new), Ordering::Release);

    self.buffer.set(new);

    std::mem::replace(buffer, new)
  }

  /// Write to the next slot, swapping buffers as necessary and returning a Stealer at the start of a new batch
  pub fn push(&self, task: T) -> Option<Stealer<T>> {
    let slot = self
      .inner
      .slot
      .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

    let mut buffer = self.buffer.get();

    // BUFFER_IDX bit changed, therefore buffer was stolen
    if ((slot ^ buffer.slot) & BUFFER_IDX).eq(&BUFFER_IDX) {
      buffer = Buffer::alloc(slot, buffer.cap);

      self
        .inner
        .get_buffer(slot)
        .store(Owned::new(buffer), Ordering::Release);

      self.buffer.set(buffer);

      unsafe {
        buffer.write(0, task);
      }

      // There can be no stealer at this point, so no need to check IDX XOR
      self
        .inner
        .slot
        .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

      let (tx, rx) = channel();
      self.tx.set(Some(tx));

      Some(Stealer::Taker(StealHandle {
        rx,
        inner: self.inner.clone(),
      }))
    } else {
      let index = slot_delta(slot, buffer.slot);

      match &self.flavor {
        Flavor::Unbounded if index.eq(&buffer.cap) => {
          unsafe {
            self.resize(&mut buffer, slot);
            buffer.write(index, task);
          }

          let slot = self
            .inner
            .slot
            .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

          // Stealer expressed intention to take buffer by changing the buffer index, and is waiting on Worker to send buffer upon completion of the current write in progress
          if ((slot ^ buffer.slot) & BUFFER_IDX).eq(&BUFFER_IDX) {
            let (tx, rx) = channel();
            let tx = self.tx.replace(Some(tx)).unwrap();

            // Send buffer as vec to receiver
            tx.send(unsafe { buffer.to_vec(index) }).ok();

            Some(Stealer::Taker(StealHandle {
              rx,
              inner: self.inner.clone(),
            }))
          } else {
            None
          }
        }
        Flavor::AutoBatched { batch_size } if index.eq(batch_size) => {
          let old = self.replace_buffer(&mut buffer, slot, *batch_size);
          let batch = unsafe { old.to_vec(*batch_size) };

          unsafe {
            buffer.write(0, task);
          }

          let slot = self
            .inner
            .slot
            .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

          if ((slot ^ buffer.slot) & BUFFER_IDX).eq(&BUFFER_IDX) {
            let (tx, rx) = channel();
            let tx = self.tx.replace(Some(tx)).unwrap();

            tx.send(batch).ok();

            Some(Stealer::Taker(StealHandle {
              rx,
              inner: self.inner.clone(),
            }))
          } else {
            Some(Stealer::Owner(batch))
          }
        }
        _ if index.eq(&0) => {
          unsafe {
            buffer.write(0, task);
          }

          self
            .inner
            .slot
            .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

          let (tx, rx) = channel();
          self.tx.set(Some(tx));

          Some(Stealer::Taker(StealHandle {
            rx,
            inner: self.inner.clone(),
          }))
        }
        _ => {
          unsafe {
            buffer.write(index, task);
          }

          let slot = self
            .inner
            .slot
            .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

          if ((slot ^ buffer.slot) & BUFFER_IDX).eq(&BUFFER_IDX) {
            let (tx, rx) = channel();
            let tx = self.tx.replace(Some(tx)).unwrap();

            // Send buffer as vec to receiver
            tx.send(unsafe { buffer.to_vec(index) }).ok();

            Some(Stealer::Taker(StealHandle {
              rx,
              inner: self.inner.clone(),
            }))
          } else {
            None
          }
        }
      }
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

impl<T> Drop for Worker<T> {
  fn drop(&mut self) {
    // By leaving this as indefinitely write in progress the Stealer will always receive from the oneshot::Sender
    let slot = self
      .inner
      .slot
      .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

    let buffer = self.buffer.get();

    // Is buffer still current? (If not Stealer has already taken buffer)
    if slot & BUFFER_IDX == buffer.slot & BUFFER_IDX {
      let length = slot_delta(slot, buffer.slot);

      // Send to Stealer if able
      if let Some(tx) = self.tx.replace(None) {
        if let Err(queue) = tx.send(unsafe { buffer.to_vec(length) }) {
          drop(queue);
        }
      } else {
        // Otherwise deallocate everything
        unsafe {
          // Go through the buffer from front to back and drop all tasks in the queue.
          for i in 0..length {
            buffer.at(i).drop_in_place();
          }

          // Free the memory allocated by the buffer.
          buffer.dealloc();
        }
      }
    }
  }
}

#[doc(hidden)]
pub struct StealHandle<T> {
  /// Buffer receiver to be used when waiting on writes
  rx: Receiver<Vec<T>>,
  /// A reference to the inner representation of the queue.
  inner: Arc<CachePadded<Inner<T>>>,
}

/// Stealers swap out and take ownership of buffers in entirety from Workers
pub enum Stealer<T> {
  /// Stealer was created with an owned batch that can simply be unwrapped
  Owner(Vec<T>),
  /// A Steal Handle buffer swaps either by taking the buffer directly or by awaiting the Worker to send on write completion
  Taker(StealHandle<T>),
}

unsafe impl<T: Send> Send for Stealer<T> {}
unsafe impl<T: Send> Sync for Stealer<T> {}

impl<T> Stealer<T> {
  /// Take the entire queue by swapping the underlying buffer and converting back into a `Vec<T>` or by waiting to receive the buffer from the Worker if a write was in progress.
  pub async fn take(self) -> Vec<T> {
    match self {
      Stealer::Owner(batch) => batch,
      Stealer::Taker(StealHandle { rx, inner }) => {
        let slot = inner.slot.fetch_xor(BUFFER_IDX, Ordering::Relaxed);

        // Worker will see the buffer has swapped when confirming length increment
        if slot & WRITE_IN_PROGRESS == WRITE_IN_PROGRESS {
          // Writer can never be dropped mid-write, therefore RecvError cannot occur
          rx.await.unwrap()
        } else {
          let guard = &epoch::pin();

          let buffer = inner.get_buffer(slot).load_consume(guard);

          unsafe {
            let buffer = *buffer.into_owned();
            buffer.to_vec(slot_delta(slot, buffer.slot))
          }
        }
      }
    }
  }

  /// Take the entire queue by swapping the underlying buffer and converting into a `Vec<T>` or by blocking to receive from the Worker if a write was in progress. This is always non-blocking when called on the same thread as the Worker
  pub fn take_blocking(self) -> Vec<T> {
    match self {
      Stealer::Owner(batch) => batch,
      Stealer::Taker(StealHandle { rx, inner }) => {
        let slot = inner.slot.fetch_xor(BUFFER_IDX, Ordering::Relaxed);

        // Worker will see the buffer has swapped when confirming length increment
        // It's not possible for this to be write in progress when called from the same thread as the queue
        if slot & WRITE_IN_PROGRESS == WRITE_IN_PROGRESS {
          // Writer can never be dropped mid-write, therefore RecvError cannot occur
          block_on(rx).unwrap()
        } else {
          let guard = &epoch::pin();

          let buffer = inner.get_buffer(slot).load_consume(guard);

          unsafe {
            let buffer = *buffer.into_owned();
            buffer.to_vec(slot_delta(slot, buffer.slot))
          }
        }
      }
    }
  }
}

/// Uses [`Stealer::take_blocking`]; non-blocking when called on the same thread as Worker
impl<T> From<Stealer<T>> for Vec<T> {
  fn from(stealer: Stealer<T>) -> Self {
    stealer.take_blocking()
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

  #[test]
  fn it_makes_new_stealer_per_batch() {
    model!({
      let queue = Worker::new();
      let stealer = queue.push(0).unwrap();

      queue.push(1);
      queue.push(2);

      assert_eq!(stealer.take_blocking(), vec![0, 1, 2]);

      let stealer = queue.push(3).unwrap();
      queue.push(4);
      queue.push(5);

      assert_eq!(stealer.take_blocking(), vec![3, 4, 5]);
    });
  }

  #[test]
  fn it_auto_batches() {
    model!({
      let queue = Worker::auto_batched(64);
      let mut stealers: Vec<Stealer<i32>> = vec![];

      for i in 0..128 {
        if let Some(stealer) = queue.push(i) {
          stealers.push(stealer);
        }
      }

      let batch: Vec<i32> = stealers
        .into_iter()
        .rev()
        .flat_map(|stealer| stealer.take_blocking())
        .collect();

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
