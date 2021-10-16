use crossbeam::{
  epoch::{self, Atomic, Owned, Shared},
  utils::{Backoff, CachePadded},
};
use std::{cell::Cell, fmt, marker::PhantomData, mem, ptr, sync::Arc};

#[cfg(loom)]
use loom::sync::atomic::{self, AtomicIsize, Ordering};

#[cfg(not(loom))]
use std::sync::atomic::{self, AtomicIsize, Ordering};

#[cfg(loom)]
use loom::thread;

#[cfg(not(loom))]
use std::thread;

// Current buffer index
const BUFFER_IDX: isize = 1 << 0;

// Marker bit to designate that the buffer has been swapped
const BUFFER_SWAPPED: isize = 1 << 1;

// Designates how many bits are set aside
const FLAGS_SHIFT: isize = 2;

// Minimum buffer capacity.
const MIN_CAP: usize = 64;

// If a buffer of at least this size is retired, thread-local garbage is flushed so that it gets
// deallocated as soon as possible.
const FLUSH_THRESHOLD_BYTES: usize = 1 << 10;

/// A buffer that holds tasks in a worker queue.
///
/// This is just a pointer to the buffer and its length - dropping an instance of this struct will
/// *not* deallocate the buffer.
struct Buffer<T> {
  /// Slot that represents the base index and buffer idx
  slot: isize,

  /// Pointer to the allocated memory.
  ptr: *mut T,

  /// Capacity of the buffer. Always a power of two.
  cap: usize,
}

unsafe impl<T> Send for Buffer<T> {}

impl<T> Buffer<T> {
  /// Allocates a new buffer with the specified capacity.
  fn alloc(slot: isize, cap: usize) -> Buffer<T> {
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
  unsafe fn at(&self, index: isize) -> *mut T {
    // `self.cap` is always a power of two.
    self.ptr.offset(index & (self.cap - 1) as isize)
  }

  /// Writes `task` into the specified `index`.
  unsafe fn write(&self, index: isize, task: T) {
    ptr::write_volatile(self.at(index), task)
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

struct Inner<T> {
  slot: AtomicIsize,
  length: AtomicIsize,
  buffers: (
    CachePadded<Atomic<Buffer<T>>>,
    CachePadded<Atomic<Buffer<T>>>,
  ),
}

impl<T> Inner<T> {
  /// Take the entire queue via swapping the underlying buffer and converting into a `Vec<T>`
  fn take_queue(&self) -> Vec<T> {
    let slot = self.slot.fetch_or(BUFFER_SWAPPED, Ordering::Relaxed);

    // Buffer already taken; no new pushes
    if slot >> FLAGS_SHIFT == 0 || slot & BUFFER_SWAPPED == BUFFER_SWAPPED {
      return vec![];
    }

    let length = slot >> FLAGS_SHIFT;

    let backoff = Backoff::new();

    // Wait for writes to catch up
    while (self.length.load(Ordering::Acquire) >> FLAGS_SHIFT).lt(&length) {
      if backoff.is_completed() {
        thread::yield_now();
      } else {
        backoff.snooze();
      }
    }

    let guard = &epoch::pin();

    let buffer = self
      .current_buffer(slot)
      .swap(Shared::null(), Ordering::Relaxed, guard);

    unsafe {
      let Buffer { slot, ptr, cap } = *buffer.as_raw();

      let length = length - (slot >> FLAGS_SHIFT);

      Vec::from_raw_parts(ptr, length as usize, cap)
    }
  }

  fn current_buffer(&self, slot: isize) -> &CachePadded<Atomic<Buffer<T>>> {
    if slot & BUFFER_IDX == 0 {
      &self.buffers.0
    } else {
      &self.buffers.1
    }
  }

  fn next_buffer(&self, slot: isize) -> &CachePadded<Atomic<Buffer<T>>> {
    if slot & BUFFER_IDX == 0 {
      &self.buffers.1
    } else {
      &self.buffers.0
    }
  }
}

impl<T> Drop for Inner<T> {
  fn drop(&mut self) {
    let slot = self.slot.load(Ordering::Relaxed);

    if slot & BUFFER_SWAPPED == 0 {
      unsafe {
        let buffer = self
          .current_buffer(slot)
          .load(Ordering::Relaxed, epoch::unprotected());

        // Go through the buffer from front to back and drop all tasks in the queue.
        for i in 0..slot {
          buffer.deref().at(i).drop_in_place();
        }

        // Free the memory allocated by the buffer.
        buffer.into_owned().into_box().dealloc();
      }
    }
  }
}

/// A worker queue.
///
/// This is a queue that is owned by a single thread, but other threads may steal the entire underlying buffer. Typically one would use a single worker queue per thread.
///
/// # Examples
///
/// ```
/// use swap_queue::Worker;
///
/// let w = Worker::new();
/// let s = w.stealer();
///
/// w.push(1);
/// w.push(2);
/// w.push(3);
/// assert_eq!(s.take_queue(), vec![1, 2, 3]);
///
/// w.push(4);
/// w.push(5);
/// w.push(6);
/// assert_eq!(s.take_queue(), vec![4, 5, 6]);
/// ```

pub struct Worker<T> {
  /// A reference to the inner representation of the queue.
  inner: Arc<CachePadded<Inner<T>>>,
  /// A copy of `inner.buffer` for quick access.
  buffer: Cell<Buffer<T>>,
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
      slot: AtomicIsize::new(0),
      length: AtomicIsize::new(0),
      buffers: (
        CachePadded::new(Atomic::new(buffer)),
        CachePadded::new(Atomic::null()),
      ),
    }));

    Worker {
      inner,
      buffer: Cell::new(buffer),
      _marker: PhantomData,
    }
  }

  /// Creates a stealer for this queue.
  ///
  /// The returned stealer can be shared among threads and cloned.
  ///
  /// # Examples
  ///
  /// ```
  /// use swap_queue::Worker;
  ///
  /// let w = Worker::<i32>::new();
  /// let s = w.stealer();
  /// ```
  pub fn stealer(&self) -> Stealer<T> {
    Stealer {
      inner: self.inner.clone(),
    }
  }

  /// Resizes the internal buffer to the new capacity of `new_cap`.
  #[cold]
  unsafe fn resize(&self, new_cap: usize, slot: isize, length: isize) {
    let buffer = self.buffer.get();

    // Allocate a new buffer and copy data from the old buffer to the new one.
    let new = Buffer::alloc(slot, new_cap);

    ptr::copy_nonoverlapping(buffer.at(0), new.at(0), length as usize);

    let guard = &epoch::pin();

    self.buffer.set(new);

    let old = self
      .inner
      .current_buffer(slot)
      .swap(Owned::new(new), Ordering::Release, guard);

    guard.defer_unchecked(move || old.into_owned().into_box().dealloc());

    // If the buffer is very large, then flush the thread-local garbage in order to deallocate
    // it as soon as possible.
    if mem::size_of::<T>() * new_cap >= FLUSH_THRESHOLD_BYTES {
      guard.flush();
    }
  }

  /// Push a task to the queue and returns the index written to
  pub fn push(&self, task: T) -> isize {
    let slot = self
      .inner
      .slot
      .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

    if slot & BUFFER_SWAPPED == BUFFER_SWAPPED {
      let buffer = Buffer::alloc(slot ^ BUFFER_IDX, MIN_CAP);

      self.buffer.set(buffer);

      self
        .inner
        .next_buffer(slot)
        .store(Owned::new(buffer), Ordering::Release);

      self
        .inner
        .slot
        .fetch_xor((1 << FLAGS_SHIFT) - 1, Ordering::Relaxed);

      unsafe {
        buffer.write(0, task);
      }

      atomic::fence(Ordering::Release);

      self
        .inner
        .length
        .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

      0
    } else {
      let mut buffer = self.buffer.get();

      let index = (slot >> FLAGS_SHIFT) - (buffer.slot >> FLAGS_SHIFT);

      // Is the queue full?
      if index >= buffer.cap as isize {
        // Yes. Grow the underlying buffer.
        unsafe {
          self.resize(2 * buffer.cap, buffer.slot, index);
        }

        buffer = self.buffer.get();
      }

      // Write `task` into the slot.
      unsafe {
        buffer.write(index, task);
      }

      atomic::fence(Ordering::Release);

      self
        .inner
        .length
        .fetch_add(1 << FLAGS_SHIFT, Ordering::Relaxed);

      index
    }
  }

  /// Take the entire queue via swapping the underlying buffer and converting into a `Vec<T>`
  pub fn take_queue(&self) -> Vec<T> {
    self.inner.take_queue()
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
pub struct Stealer<T> {
  /// A reference to the inner representation of the queue.
  inner: Arc<CachePadded<Inner<T>>>,
}

unsafe impl<T: Send> Send for Stealer<T> {}
unsafe impl<T: Send> Sync for Stealer<T> {}

impl<T> Stealer<T> {
  /// Take the entire queue via swapping the underlying buffer and converting into a `Vec<T>`
  pub fn take_queue(&self) -> Vec<T> {
    self.inner.take_queue()
  }
}

impl<T> Clone for Stealer<T> {
  fn clone(&self) -> Stealer<T> {
    Stealer {
      inner: self.inner.clone(),
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
  use loom::{sync::mpsc::channel, thread};

  #[cfg(not(loom))]
  use std::{sync::mpsc::channel, thread};

  macro_rules! model {
    ($test:block) => {
      #[cfg(loom)]
      loom::model(|| $test);

      #[cfg(not(loom))]
      $test
    };
  }

  #[test]
  fn multi_stealer() {
    model!({
      let queue = Worker::new();
      let stealer = queue.stealer();

      for i in 0..50 {
        queue.push(i);
      }

      let threads = (0..2).map(|_| {
        let stealer = stealer.clone();
        thread::spawn(move || {
          let batch = stealer.take_queue();
          batch.len()
        })
      });

      let received_len: std::thread::Result<usize> =
        threads.into_iter().map(|thread| thread.join()).sum();

      assert_eq!(received_len.unwrap(), 50);
    });
  }

  #[test]
  fn it_resizes() {
    model!({
      let queue = Worker::new();

      for i in 0..256 {
        queue.push(i);
      }

      let batch = queue.take_queue();
      let expected = (0..256).collect::<Vec<i32>>();

      assert_eq!(batch, expected);
    });
  }

  #[test]
  fn stealer_takes() {
    model!({
      let queue = Worker::new();
      let stealer = queue.stealer();

      for i in 0..100 {
        queue.push(i);
      }

      thread::spawn(move || {
        stealer.take_queue();
      })
      .join()
      .unwrap();
    });
  }

  #[test]
  fn multi_steals() {
    model!({
      let queue = Worker::new();
      let stealer = queue.stealer();

      for i in 0..128 {
        queue.push(i);
      }

      let mut batch = stealer.take_queue();

      for i in 128..256 {
        queue.push(i);
      }

      batch.extend(stealer.take_queue());

      assert_eq!(batch, (0..256).collect::<Vec<i32>>());
    });
  }

  #[test]
  fn takes_while_pushing() {
    model!({
      let (tx, rx) = channel::<Stealer<i32>>();

      let handles = vec![
        thread::spawn(move || {
          let queue = Worker::new();

          for i in 0..64 {
            queue.push(i);
          }

          let stealer = queue.stealer();
          tx.send(stealer).unwrap();

          for i in 0..64 {
            queue.push(i);
          }
        }),
        thread::spawn(move || {
          let stealer = rx.recv().unwrap();
          stealer.take_queue();
        }),
      ];

      for handle in handles {
        handle.join().unwrap();
      }
    });
  }
}
