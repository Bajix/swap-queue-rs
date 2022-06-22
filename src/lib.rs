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
  future::Future,
  hint,
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

// Marker flag to designate that the queue has never been used
// This works because base_slot is overidden by slot due to the initial lane being out of phase
const QUEUE_UNUSED: usize = 1 << 3;

// Stealer dropped while uninitialized and tasks need to be dropped
const DROP_BUFFER: usize = LANE | RECLAMATION_PHASE;

// Designates that write is in progress
const WRITE_IN_PROGRESS: usize = 1 << 31;

// Designates that receiver value was previous set; flag does not reset when value taken
const VALUE_SET: usize = 1 << 0;

// Designates async waker was previously set; flag does not reset when waker taken
const WAKER_REGISTERED: usize = 1 << 2;

// Designates receiver value already set during drop
const VALUE_RECLAIMED: usize = RECLAMATION_PHASE | VALUE_SET;
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
unsafe impl<T: Send> Sync for Buffer<T> {}

impl<T> Buffer<T> {
  /// Allocates a new buffer with the specified capacity.
  #[inline]
  fn alloc(cap: usize) -> Buffer<T> {
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
  unsafe fn at(&self, index: isize) -> *mut T {
    self.ptr.offset(index)
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

struct RawVec<T> {
  buffer: Buffer<T>,
  len: usize,
}

impl<T> RawVec<T> {
  #[inline]
  unsafe fn to_vec(self) -> Vec<T> {
    self.buffer.to_vec(self.len)
  }

  // Bitwise copy data from array pointer into raw buffer while resizing as necessary
  unsafe fn extend(&mut self, lane: *const MaybeUninit<T>, count: usize) {
    if (self.len + count) > self.buffer.cap {
      let new = Buffer::alloc(self.buffer.cap * 2);

      ptr::copy_nonoverlapping(self.buffer.at(0), new.at(0), self.buffer.cap);

      let old = mem::replace(&mut self.buffer, new);

      // Here we're deallocating the old buffer without dropping individual items as the new buffer is now owner
      old.dealloc();
    }

    ptr::copy_nonoverlapping(
      lane.offset(0).cast(),
      self.buffer.at(self.len as isize),
      count,
    );

    self.len += count;
  }
}

#[inline]
fn slot_delta(a: &usize, b: &usize) -> usize {
  ((a >> 32) as u32).wrapping_sub((b >> 32) as u32) as usize
}
struct Inner<T: Sized, const N: usize> {
  slot: CachePadded<AtomicUsize>,
  base_slot: CachePadded<UnsafeCell<usize>>,
  buffers: ([MaybeUninit<T>; N], [MaybeUninit<T>; N]),
}

impl<T: Sized, const N: usize> Inner<T, N> {
  fn new() -> Self {
    Inner {
      // Lane is always initially out of phase to trigger issuance of initial Stealer
      // QUEUE_UNUSED gets overriden on initial push
      base_slot: CachePadded::new(UnsafeCell::new(LANE | QUEUE_UNUSED)),
      slot: CachePadded::new(AtomicUsize::new(0)),
      buffers: unsafe {
        (
          MaybeUninit::uninit().assume_init(),
          MaybeUninit::uninit().assume_init(),
        )
      },
    }
  }

  #[inline]
  fn buffer(&self, slot: &usize) -> *const MaybeUninit<T> {
    if slot & LANE == 0 {
      self.buffers.0.as_ptr()
    } else {
      self.buffers.1.as_ptr()
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

pub struct SwapQueue<T: Send + Sized, const N: usize> {
  /// The inner representation of the queue.
  inner: Inner<T, N>,
  /// Value receiver owned by current Stealer
  rx: UnsafeCell<Option<Arc<Receiver<T>>>>,
  /// Indicate that queue is !Sync; only a single thread can safely push
  _marker: PhantomData<Cell<T>>,
}

unsafe impl<T: Send, const N: usize> Send for SwapQueue<T, N> {}

impl<T: Send + Sized, const N: usize> SwapQueue<T, N> {
  /// Creates a new SwapQueue queue.
  ///
  /// # Examples
  ///
  /// ```
  /// use swap_queue::SwapQueue;
  ///
  /// let queue = SwapQueue::<i32>::new();
  /// ```
  pub fn new() -> SwapQueue<T, N> {
    debug_assert_eq!(N, N.next_power_of_two());

    SwapQueue {
      inner: Inner::new(),
      rx: UnsafeCell::new(None),
      _marker: PhantomData,
    }
  }

  unsafe fn take_rx(&self) -> Option<Arc<Receiver<T>>> {
    mem::replace(&mut *self.rx.get(), None)
  }

  /// Write to the next slot, returning a Stealer at the start of a new batch
  pub fn push(&self, task: T) -> Option<Stealer<T, N>> {
    let slot = self
      .inner
      .slot
      .fetch_add(WRITE_IN_PROGRESS, Ordering::Relaxed);

    let base_slot = unsafe { *self.inner.base_slot.get() };

    let lane = self.inner.buffer(&slot);

    if ((slot ^ base_slot) & LANE).eq(&LANE) {
      unsafe {
        ptr::write(lane.offset(0) as *mut _, MaybeUninit::new(task));
      }

      unsafe {
        *self.inner.base_slot.get() = slot;
      }

      self
        .inner
        .slot
        .fetch_add(WRITE_IN_PROGRESS, Ordering::Relaxed);

      let stealer = Stealer::new(ptr::addr_of!(self.inner), slot);

      unsafe {
        self.rx.get().replace(Some(stealer.rx.clone()));
      }

      Some(stealer)
    } else {
      let index = slot_delta(&slot, &base_slot);
      // This works because N is always a power of 2
      let offset = index & (N - 1);

      // Write slot has wrapped around to the beginning
      if offset == 0 {
        unsafe {
          // Move data into receiver value by way of bitwise copy
          (*self.rx.get())
            .as_ref()
            .unwrap()
            .splice_unchecked::<N>(lane, index - N, N);
        }
      }

      unsafe {
        ptr::write(
          lane.offset(offset as isize) as *mut _,
          MaybeUninit::new(task),
        );
      }

      let slot = self
        .inner
        .slot
        .fetch_add(WRITE_IN_PROGRESS, Ordering::Relaxed);

      match (slot ^ base_slot) & (LANE | RECLAMATION_PHASE) {
        // Stealer dropped
        DROP_BUFFER => {
          let length = index + 1;

          let count = {
            let offset = length & (N - 1);

            if offset.eq(&0) {
              N
            } else {
              offset
            }
          };

          // Go through the buffer from front to back and drop all tasks in the queue.
          for i in 0..count {
            unsafe {
              (lane.offset(i as isize) as *mut T).drop_in_place();
            }
          }

          if let Some(rx) = unsafe { self.take_rx() } {
            if length > N {
              drop(unsafe { rx.take_value() });
            }
          }
        }
        // Stealer waiting to wake / take
        LANE => unsafe {
          let rx = self.take_rx().unwrap();
          if index < N {
            rx.set_value::<N>(lane);
          } else {
            let length = index + 1;

            let count = {
              let offset = length & (N - 1);

              if offset.eq(&0) {
                N
              } else {
                offset
              }
            };

            rx.splice_unchecked::<N>(lane, length - count, count);
          }
          rx.take_waker().wake();
        },
        _ => (),
      }

      None
    }
  }
}

impl<T: Send + Sized, const N: usize> Default for SwapQueue<T, N> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: Send + Sized, const N: usize> Drop for SwapQueue<T, N> {
  fn drop(&mut self) {
    // If there is no receiver, there's no data to move or drop
    if let Some(rx) = unsafe { self.take_rx() } {
      // Stealer has to touch rx state first to know if value is or will be set
      // Once RECLAMATION_PHASE has been observed, the stealer inner raw pointer isn't valid
      let rx_state = rx.state.fetch_or(RECLAMATION_PHASE, Ordering::Relaxed);

      if (rx_state & RECLAMATION_PHASE).eq(&RECLAMATION_PHASE) {
        // Guarantee pointer is valid while stealer is dropping data
        while Arc::strong_count(&rx).gt(&1) {
          hint::spin_loop();
        }

        return;
      }

      let slot = self
        .inner
        .slot
        .fetch_xor(RECLAMATION_PHASE, Ordering::Relaxed);

      let base_slot = unsafe { *self.inner.base_slot.get() };

      if ((slot ^ base_slot) & LANE).eq(&LANE) {
        // Guarantee pointer is valid while stealer is moving/dropping data
        while Arc::strong_count(&rx).gt(&1) {
          hint::spin_loop();
        }
      } else {
        let lane = self.inner.buffer(&base_slot);
        let length = slot_delta(&slot, &base_slot);

        unsafe { rx.append_unchecked::<N>(lane, length) }

        let rx_state = rx.state.fetch_or(VALUE_SET, Ordering::Relaxed);

        // Stealer observed RECLAMATION_PHASE already and needs to be awoken if it registered before VALUE_SET was set
        if (rx_state & WAKER_REGISTERED).eq(&WAKER_REGISTERED) {
          unsafe {
            rx.take_waker().wake();
          }
        }
      }
    }
  }
}

struct Receiver<T> {
  state: AtomicUsize,
  value: UnsafeCell<MaybeUninit<RawVec<T>>>,
  waker: UnsafeCell<MaybeUninit<Waker>>,
}

unsafe impl<T: Send> Send for Receiver<T> {}
unsafe impl<T: Send> Sync for Receiver<T> {}

impl<T> Receiver<T> {
  fn new() -> Self {
    Receiver {
      state: AtomicUsize::new(0),
      value: UnsafeCell::new(MaybeUninit::uninit()),
      waker: UnsafeCell::new(MaybeUninit::uninit()),
    }
  }

  // Bitwise copy data from array pointer into Receiver value
  unsafe fn set_value<const N: usize>(&self, lane: *const MaybeUninit<T>) {
    let buffer: Buffer<T> = Buffer::alloc(N);

    ptr::copy_nonoverlapping(lane.offset(0).cast(), buffer.at(0), N);

    self
      .value
      .get()
      .write(MaybeUninit::new(RawVec { buffer, len: N }));
  }

  unsafe fn take_value(&self) -> RawVec<T> {
    mem::replace(&mut *self.value.get(), MaybeUninit::uninit()).assume_init()
  }

  unsafe fn set_waker(&self, waker: Waker) {
    self.waker.get().write(MaybeUninit::new(waker));
  }

  unsafe fn take_waker(&self) -> Waker {
    mem::replace(&mut *self.waker.get(), MaybeUninit::uninit()).assume_init()
  }

  // Bitwise copy data from array pointer into Receiver value at index while resizing as necessary
  unsafe fn splice_unchecked<const N: usize>(
    &self,
    lane: *const MaybeUninit<T>,
    start_index: usize,
    count: usize,
  ) {
    if start_index == 0 {
      let buffer: Buffer<T> = Buffer::alloc(N * 2);

      ptr::copy_nonoverlapping(lane.offset(0).cast(), buffer.at(0), count);

      self
        .value
        .get()
        .write(MaybeUninit::new(RawVec { buffer, len: count }));
    } else {
      (*self.value.get()).assume_init_mut().extend(lane, count);
    }
  }

  // Bitwise copy data from array pointer into Receiver value relative to final length
  unsafe fn append_unchecked<const N: usize>(&self, lane: *const MaybeUninit<T>, length: usize) {
    if length.le(&N) {
      self.set_value::<N>(lane);
    } else {
      let count = {
        let offset = length & (N - 1);
        if offset.eq(&0) {
          N
        } else {
          offset
        }
      };

      self.splice_unchecked::<N>(lane, length - count, count);
    }
  }
}

pub enum State {
  Uninitialized,
  Pending,
  Received,
}
pub struct Stealer<T, const N: usize> {
  /// A raw pointer to the stack allocated inner representation of the queue.
  ptr: *const Inner<T, N>,
  /// Current polling state
  state: State,
  /// Slot that represents the base index offset and buffer idx / reclamation phase
  base_slot: usize,
  /// Receiver for value exchange
  rx: Arc<Receiver<T>>,
}

unsafe impl<T: Send, const N: usize> Send for Stealer<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for Stealer<T, N> {}

impl<T, const N: usize> Stealer<T, N> {
  fn new(ptr: *const Inner<T, N>, base_slot: usize) -> Self {
    Stealer {
      ptr,
      base_slot,
      state: State::Uninitialized,
      rx: Arc::new(Receiver::new()),
    }
  }

  unsafe fn inner(&self) -> &Inner<T, N> {
    self.ptr.as_ref().expect("Stealer ptr invalid")
  }
}

impl<T, const N: usize> Future for Stealer<T, N> {
  type Output = Vec<T>;
  fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
    match self.state {
      State::Pending => {
        let value = unsafe { self.rx.take_value().to_vec() };

        unsafe {
          self.get_unchecked_mut().state = State::Received;
        }

        Poll::Ready(value)
      }
      State::Uninitialized => {
        unsafe {
          self.rx.set_waker(cx.waker().clone());
        }

        let rx_state = self.rx.state.fetch_or(WAKER_REGISTERED, Ordering::Relaxed);

        match rx_state & (RECLAMATION_PHASE | VALUE_SET) {
          VALUE_RECLAIMED | VALUE_SET => {
            let value = unsafe { self.rx.take_value().to_vec() };

            unsafe {
              self.get_unchecked_mut().state = State::Received;
            }

            Poll::Ready(value)
          }
          RECLAMATION_PHASE => {
            unsafe {
              self.get_unchecked_mut().state = State::Pending;
            }

            Poll::Pending
          }
          _ => {
            let slot = unsafe { self.inner().slot.fetch_xor(LANE, Ordering::Relaxed) };

            if (slot & WRITE_IN_PROGRESS).eq(&WRITE_IN_PROGRESS) {
              unsafe {
                self.get_unchecked_mut().state = State::Pending;
              }

              Poll::Pending
            } else {
              unsafe {
                drop(self.rx.take_waker());
              }

              let length = slot_delta(&slot, &self.base_slot);

              let value = if length.le(&N) {
                unsafe {
                  let buffer: Buffer<T> = Buffer::alloc(length);
                  ptr::copy_nonoverlapping(
                    self.inner().buffer(&slot).offset(0).cast(),
                    buffer.ptr.offset(0),
                    length,
                  );
                  buffer.to_vec(length)
                }
              } else {
                unsafe {
                  let mut raw = self.rx.take_value();

                  let count = {
                    let offset = length & (N - 1);
                    if offset.eq(&0) {
                      N
                    } else {
                      offset
                    }
                  };

                  raw.extend(self.inner().buffer(&slot), count);
                  raw.to_vec()
                }
              };

              unsafe {
                self.get_unchecked_mut().state = State::Received;
              }

              Poll::Ready(value)
            }
          }
        }
      }
      State::Received => {
        unreachable!()
      }
    }
  }
}

impl<T, const N: usize> Drop for Stealer<T, N> {
  fn drop(&mut self) {
    match self.state {
      // It should not be possible for Stealer to drop while pending; wake is imminent.
      State::Pending => {
        unreachable!()
      }
      State::Uninitialized => {
        let rx_state = self.rx.state.fetch_or(RECLAMATION_PHASE, Ordering::Relaxed);

        if (rx_state & RECLAMATION_PHASE).eq(&0) {
          unsafe {
            let slot = self
              .inner()
              .slot
              .fetch_xor(LANE | RECLAMATION_PHASE, Ordering::Relaxed);

            if (slot & WRITE_IN_PROGRESS).eq(&0) {
              let length = slot_delta(&slot, &self.base_slot);
              let count = {
                let offset = length & (N - 1);
                if offset.eq(&0) {
                  N
                } else {
                  offset
                }
              };

              let buffer = self.inner().buffer(&slot) as *mut T;

              // Go through the buffer from front to back and drop all tasks in the queue.
              for i in 0..count {
                buffer.offset(i as isize).drop_in_place();
              }

              if length > N {
                drop(self.rx.take_value().to_vec());
              }
            }
          }
        }
      }
      State::Received => {}
    }
  }
}

#[cfg(all(test))]
mod tests {
  use super::*;

  #[test]
  fn raw_vec_extends() {
    let mut raw_vec: RawVec<u64> = RawVec {
      buffer: Buffer::alloc(64),
      len: 0,
    };

    let values: [u64; 32] = Default::default();

    unsafe {
      raw_vec.extend(values.as_ptr().cast(), 32);
      raw_vec.extend(values.as_ptr().cast(), 16);
      raw_vec.extend(values.as_ptr().cast(), 32);
      raw_vec.extend(values.as_ptr().cast(), 16);
    }

    assert_eq!(
      unsafe { raw_vec.to_vec() },
      std::iter::repeat(0).take(96).collect::<Vec<u64>>()
    );
  }

  #[test]
  fn slot_wraps_around() {
    let delta = slot_delta(&0, &((u32::MAX as usize) << 32));

    assert_eq!(delta, 1);
  }

  #[tokio::test]
  async fn it_resizes() {
    let queue: SwapQueue<i32, 64> = SwapQueue::new();
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
    let queue: SwapQueue<i32, 64> = SwapQueue::new();
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
  async fn stress_test() {
    let queue: SwapQueue<i32, 64> = SwapQueue::new();

    for i in 1..1048 {
      let stealer = queue.push(0).unwrap();

      (1..i).for_each(|x| {
        queue.push(x);
      });

      assert_eq!(stealer.await, (0..i).into_iter().collect::<Vec<i32>>());
    }
  }

  #[tokio::test]
  async fn stealer_takes() {
    let queue: SwapQueue<i32, 64> = SwapQueue::new();
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
    let queue: SwapQueue<i32, 64> = SwapQueue::new();
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
