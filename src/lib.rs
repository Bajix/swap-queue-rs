//!
//! A lock-free thread-owned queue whereby tasks are taken by stealers in entirety via buffer swapping. This is meant to be used [`thread_local`] paired with [`tokio::task::spawn`] as a constant-time take-all batching mechanism that outperforms [`crossbeam_deque::SwapQueue`](https://docs.rs/crossbeam-deque/0.8.1/crossbeam_deque/struct.SwapQueue.html), and [`tokio::sync::mpsc`] for batching.
//!
//! ## Example
//!
//! ```
//! use swap_queue::SwapQueue;
//! use tokio::{
//!   runtime::Handle,
//!   sync::oneshot::{channel, Sender},
//! };
//!
//! // Jemalloc makes this library substantially faster
//! #[global_allocator]
//! static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
//!
//! // SwapQueue needs to be thread local because it is !Sync
//! thread_local! {
//!   static QUEUE: SwapQueue<(u64, Sender<u64>)> = SwapQueue::new();
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
//!           let batch = stealer.await;
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

use cache_padded::CachePadded;
use std::{
  cell::{Cell, UnsafeCell},
  fmt,
  future::Future,
  marker::PhantomData,
  mem::{self, MaybeUninit},
  pin::Pin,
  ptr,
  sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
  },
  task::{Context, Poll, Waker},
};

// Current buffer index
const LANE: usize = 1 << 0;

// Phase for determining reclamation
const RECLAMATION_PHASE: usize = 1 << 1;

const DROP_BUFFER: usize = LANE | RECLAMATION_PHASE;

// Designates that write is in progress
const WRITE_IN_PROGRESS: usize = 1 << 31;

// Minimum buffer capacity.
const MIN_CAP: usize = 64;

/// A buffer that holds tasks in a worker queue.
///
/// This is just a pointer to the buffer and its length - dropping an instance of this struct will
/// *not* deallocate the buffer.
struct Buffer<T> {
  /// Pointer to the allocated memory.
  ptr: *mut T,

  /// Capacity of the buffer. Always a power of two.
  cap: usize,
}

unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}

impl<T> Buffer<T> {
  /// Allocates a new buffer with the specified capacity.
  #[inline]
  fn alloc(cap: usize) -> Buffer<T> {
    debug_assert_eq!(cap, cap.next_power_of_two());

    let mut v = Vec::with_capacity(cap);
    let ptr = v.as_mut_ptr();
    mem::forget(v);

    Buffer { ptr, cap }
  }

  /// Deallocates the buffer.
  #[inline]
  unsafe fn dealloc(self) {
    drop(Vec::from_raw_parts(self.ptr, 0, self.cap));
  }

  /// Returns a pointer to the task at the specified `index`.
  #[inline]
  unsafe fn at(&self, index: usize) -> *mut T {
    // `self.cap` is always a power of two.
    self.ptr.offset((index & (self.cap - 1)) as isize)
  }

  /// Writes `task` into the specified `index`.
  #[inline]
  unsafe fn write(&self, index: usize, task: T) {
    ptr::write_volatile(self.at(index), task)
  }

  #[inline]
  unsafe fn to_vec(self, length: usize) -> Vec<T> {
    let Buffer { ptr, cap, .. } = self;
    Vec::from_raw_parts(ptr, length, cap)
  }
}

impl<T> Clone for Buffer<T> {
  fn clone(&self) -> Buffer<T> {
    Buffer {
      ptr: self.ptr,
      cap: self.cap,
    }
  }
}

#[inline]
fn slot_delta(a: &usize, b: &usize) -> usize {
  ((a >> 32) as u32).wrapping_sub((b >> 32) as u32) as usize
}

struct Inner<T> {
  slot: CachePadded<AtomicUsize>,
  base_slot: CachePadded<UnsafeCell<usize>>,
  buffers: (
    UnsafeCell<MaybeUninit<Buffer<T>>>,
    UnsafeCell<MaybeUninit<Buffer<T>>>,
  ),
  waker: UnsafeCell<MaybeUninit<Waker>>,
}

impl<T> Inner<T> {
  #[inline]
  fn buffer(&self, slot: &usize) -> &UnsafeCell<MaybeUninit<Buffer<T>>> {
    if slot & LANE == 0 {
      &self.buffers.0
    } else {
      &self.buffers.1
    }
  }

  #[inline]
  unsafe fn take_buffer(&self, slot: &usize) -> Buffer<T> {
    mem::replace(&mut *self.buffer(slot).get(), MaybeUninit::uninit()).assume_init()
  }

  #[inline]
  unsafe fn take_waker(&self) -> Waker {
    mem::replace(&mut *self.waker.get(), MaybeUninit::uninit()).assume_init()
  }

  /// Resizes the internal buffer to the new capacity.
  unsafe fn resize_if_necessary(
    &self,
    write_index: &usize,
    buffer_cell: &UnsafeCell<MaybeUninit<Buffer<T>>>,
  ) {
    let current_cap = (*buffer_cell.get()).assume_init_ref().cap;

    if current_cap.eq(write_index) {
      // Allocate a new buffer and copy data from the old buffer to the new one.
      let new = Buffer::alloc(current_cap * 2);

      let buffer: &mut Buffer<T> = { (*buffer_cell.get()).assume_init_mut() };

      ptr::copy_nonoverlapping(buffer.at(0), new.at(0), current_cap);

      let old = mem::replace(buffer, new);

      // Here we're deallocating the old buffer without dropping individual items as the new buffer is now owner
      old.dealloc();
    }
  }

  // This is safe so long as only called from one thread (the owner of SwapQueue)
  unsafe fn push(self: &Arc<Self>, task: T) -> Option<Stealer<T>> {
    let slot = self.slot.fetch_add(WRITE_IN_PROGRESS, Ordering::Relaxed);
    let buffer_cell = self.buffer(&slot);

    // Buffer was stolen or hasn't been initialized
    if ((slot ^ &*self.base_slot.get()) & LANE).eq(&LANE) {
      let buffer = Buffer::alloc(MIN_CAP);

      buffer.write(0, task);
      buffer_cell.get().write(MaybeUninit::new(buffer));

      *self.base_slot.get() = slot;

      self.slot.fetch_add(WRITE_IN_PROGRESS, Ordering::Relaxed);

      Some(Stealer::new(slot, self.clone()))
    } else {
      let index = slot_delta(&slot, &*self.base_slot.get());

      self.resize_if_necessary(&index, buffer_cell);
      (*buffer_cell.get()).assume_init_ref().write(index, task);

      let slot = self.slot.fetch_add(WRITE_IN_PROGRESS, Ordering::Relaxed);

      match (slot ^ &*self.base_slot.get()) & (LANE | RECLAMATION_PHASE) {
        // Stealer dropped
        DROP_BUFFER => {
          let buffer = mem::replace(&mut *buffer_cell.get(), MaybeUninit::uninit()).assume_init();

          // Go through the buffer from front to back and drop all tasks in the queue.
          for i in 0..index {
            buffer.at(i).drop_in_place();
          }

          // Free the memory allocated by the buffer.
          buffer.dealloc();

          None
        }
        // Stealer waiting to wake / take
        LANE => {
          self.take_waker().wake();

          None
        }
        _ => None,
      }
    }
  }
}

/// A thread-owned worker queue that writes to a swappable buffer using atomic slotting
///
/// # Examples
///
/// ```
/// use swap_queue::SwapQueue;
///
/// let w = SwapQueue::new();
/// let s = w.push(1).unwrap();
/// w.push(2);
/// w.push(3);
/// // this is non-blocking because it's called on the same thread as SwapQueue; a write in progress is not possible
/// assert_eq!(s.take_blocking(), vec![1, 2, 3]);
///
/// let s = w.push(4).unwrap();
/// w.push(5);
/// w.push(6);
/// // this is identical to [`Stealer::take_blocking`]
/// let batch: Vec<_> = s.into();
/// assert_eq!(batch, vec![4, 5, 6]);
/// ```

pub struct SwapQueue<T: Sized> {
  /// A reference to the inner representation of the queue.
  inner: Arc<Inner<T>>,
  /// Indicate that queue is !Sync; only a single thread can safely push
  _marker: PhantomData<Cell<T>>,
}

unsafe impl<T: Send> Send for SwapQueue<T> {}

impl<T: Sized> SwapQueue<T> {
  /// Creates a new SwapQueue queue.
  ///
  /// # Examples
  ///
  /// ```
  /// use swap_queue::SwapQueue;
  ///
  /// let queue = SwapQueue::<i32>::new();
  /// ```
  pub fn new() -> SwapQueue<T> {
    let inner = Arc::new(Inner {
      // Whenever LANE is out of sync during a push, a buffer is allocated and a stealer issued
      base_slot: CachePadded::new(UnsafeCell::new(LANE)),
      slot: CachePadded::new(AtomicUsize::new(0)),
      buffers: (
        UnsafeCell::new(MaybeUninit::uninit()),
        UnsafeCell::new(MaybeUninit::uninit()),
      ),
      waker: UnsafeCell::new(MaybeUninit::uninit()),
    });

    SwapQueue {
      inner,
      _marker: PhantomData,
    }
  }

  /// Write to the next slot, swapping buffers as necessary and returning a Stealer at the start of a new batch
  pub fn push(&self, task: T) -> Option<Stealer<T>> {
    unsafe { self.inner.push(task) }
  }
}

impl<T> Default for SwapQueue<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T> fmt::Debug for SwapQueue<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.pad("SwapQueue { .. }")
  }
}

pub enum State {
  Uninitialized,
  Pending { slot: usize },
  Received,
}
pub struct Stealer<T> {
  /// Current polling state
  state: State,
  /// Slot that represents the base index offset and buffer idx / reclamation phase
  base_slot: usize,
  /// A pointer to the inner representation of the queue.
  inner: Arc<Inner<T>>,
}

unsafe impl<T: Send> Send for Stealer<T> {}
unsafe impl<T: Sync> Sync for Stealer<T> {}

impl<T> Stealer<T> {
  #[inline]
  fn new(base_slot: usize, inner: Arc<Inner<T>>) -> Self {
    Stealer {
      base_slot,
      inner,
      state: State::Uninitialized,
    }
  }
}

impl<T> Future for Stealer<T> {
  type Output = Vec<T>;
  fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
    let inner = self.inner.as_ref();

    match self.state {
      State::Pending { slot } => {
        let value = unsafe {
          let buffer = inner.take_buffer(&slot);
          let length = slot_delta(&slot, &self.base_slot);
          buffer.to_vec(length)
        };

        unsafe {
          self.get_unchecked_mut().state = State::Received;
        }
        Poll::Ready(value)
      }
      State::Uninitialized => {
        unsafe {
          inner
            .waker
            .get()
            .write(MaybeUninit::new(cx.waker().clone()));
        }

        let slot = inner.slot.fetch_xor(LANE, Ordering::Relaxed);

        if (slot & WRITE_IN_PROGRESS).eq(&WRITE_IN_PROGRESS) {
          unsafe {
            self.get_unchecked_mut().state = State::Pending { slot };
          }

          Poll::Pending
        } else {
          unsafe {
            drop(inner.take_waker());
          }

          let value = unsafe {
            let buffer = inner.take_buffer(&slot);
            let length = slot_delta(&slot, &self.base_slot);
            buffer.to_vec(length)
          };

          unsafe {
            self.get_unchecked_mut().state = State::Received;
          }

          Poll::Ready(value)
        }
      }
      State::Received => {
        unreachable!()
      }
    }
  }
}

impl<T> Drop for Stealer<T> {
  fn drop(&mut self) {
    match self.state {
      // It should not be possible for Stealer to drop while pending; wake is imminent.
      State::Pending { slot: _ } => {
        unreachable!()
      }
      State::Uninitialized => {
        let slot = self
          .inner
          .slot
          .fetch_xor(LANE | RECLAMATION_PHASE, Ordering::Relaxed);

        // If there is no write in progress, immediately deallocate
        if (slot & WRITE_IN_PROGRESS).eq(&0) {
          let index = slot_delta(&slot, &self.base_slot);

          unsafe {
            let buffer = self.inner.take_buffer(&slot);

            // Go through the buffer from front to back and drop all tasks in the queue.
            for i in 0..index {
              buffer.at(i).drop_in_place();
            }

            // Free the memory allocated by the buffer.
            buffer.dealloc();
          }
        }
      }
      State::Received => {}
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

  #[test]
  fn slot_wraps_around() {
    let delta = slot_delta(&0, &((u32::MAX as usize) << 32));

    assert_eq!(delta, 1);
  }

  #[tokio::test]
  async fn it_resizes() {
    let queue = SwapQueue::new();
    let stealer = queue.push(0).unwrap();

    for i in 1..128 {
      queue.push(i);
    }

    let batch = stealer.await;
    let expected = (0..128).collect::<Vec<i32>>();

    assert_eq!(batch, expected);
  }

  #[tokio::test]
  async fn it_makes_new_stealer_per_batch() {
    let queue = SwapQueue::new();
    let stealer = queue.push(0).unwrap();

    queue.push(1);
    queue.push(2);

    assert_eq!(stealer.await, vec![0, 1, 2]);

    let stealer = queue.push(3).unwrap();
    queue.push(4);
    queue.push(5);

    assert_eq!(stealer.await, vec![3, 4, 5]);
  }

  #[tokio::test]
  async fn stealer_takes() {
    let queue = SwapQueue::new();
    let stealer = queue.push(0).unwrap();

    for i in 1..1024 {
      queue.push(i);
    }

    let batch = stealer.await;
    let expected = (0..1024).collect::<Vec<i32>>();

    assert_eq!(batch, expected);
  }

  #[tokio::test]
  async fn queue_drops() {
    let queue = SwapQueue::new();
    let stealer = queue.push(0).unwrap();

    for i in 1..128 {
      queue.push(i);
    }

    drop(queue);

    let batch = stealer.await;
    let expected = (0..128).collect::<Vec<i32>>();

    assert_eq!(batch, expected);
  }
}
