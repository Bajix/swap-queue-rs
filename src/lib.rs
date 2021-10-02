use crossbeam::{
  epoch::{self, Atomic, Owned},
  utils::CachePadded,
};
use std::{
  cell::Cell,
  fmt,
  marker::PhantomData,
  mem, ptr,
  sync::{
    atomic::{self, Ordering},
    Arc,
  },
};

#[cfg(loom)]
pub(crate) use loom::sync::atomic::AtomicIsize;

#[cfg(not(loom))]
pub(crate) use std::sync::atomic::AtomicIsize;

// Marker flag to designate that the buffer has already been swapped
const BUFFER_SWAPPED: isize = 1 << 0;

// Designates how many bits are set aside for
const FLAGS_SHIFT: isize = 1;

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
  /// Pointer to the allocated memory.
  ptr: *mut T,

  /// Capacity of the buffer. Always a power of two.
  cap: usize,
}

unsafe impl<T> Send for Buffer<T> {}

impl<T> Buffer<T> {
  /// Allocates a new buffer with the specified capacity.
  fn alloc(cap: usize) -> Buffer<T> {
    debug_assert_eq!(cap, cap.next_power_of_two());

    let mut v = Vec::with_capacity(cap);
    let ptr = v.as_mut_ptr();
    mem::forget(v);

    Buffer { ptr, cap }
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
      ptr: self.ptr,
      cap: self.cap,
    }
  }
}

impl<T> Copy for Buffer<T> {}

struct Inner<T> {
  slot: AtomicIsize,
  buffer: CachePadded<Atomic<Buffer<T>>>,
}

impl<T> Drop for Inner<T> {
  fn drop(&mut self) {
    let slot = self.slot.load(Ordering::Relaxed);
    let slot = slot >> FLAGS_SHIFT;

    unsafe {
      let buffer = self.buffer.load(Ordering::Relaxed, epoch::unprotected());

      // Go through the buffer from front to back and drop all tasks in the queue.
      for i in 0..slot {
        buffer.deref().at(i).drop_in_place();
      }

      // Free the memory allocated by the buffer.
      buffer.into_owned().into_box().dealloc();
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
///
/// assert_eq!(s.take_queue(), vec![1, 2, 3]);
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
    let buffer = Buffer::alloc(MIN_CAP);

    let inner = Arc::new(CachePadded::new(Inner {
      slot: AtomicIsize::new(0),
      buffer: CachePadded::new(Atomic::new(buffer)),
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
  unsafe fn resize(&self, new_cap: usize) {
    let slot = self.inner.slot.load(Ordering::Relaxed);
    let slot = slot >> FLAGS_SHIFT;

    let guard = &epoch::pin();

    let buffer = self
      .inner
      .buffer
      .load(Ordering::Relaxed, guard)
      .as_ref()
      .unwrap();

    // Allocate a new buffer and copy data from the old buffer to the new one.
    let new = Buffer::alloc(new_cap);
    for i in 0..slot {
      ptr::copy_nonoverlapping(buffer.at(i), new.at(i), 1);
    }

    let guard = &epoch::pin();

    self.buffer.replace(new);

    let old = self
      .inner
      .buffer
      .swap(Owned::new(new).into_shared(guard), Ordering::Release, guard);

    // Destroy the old buffer later.
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
      self
      .inner
      .slot
      .fetch_sub(1 << FLAGS_SHIFT, Ordering::Relaxed);
      let guard = &epoch::pin();

      // Replacement was already swapped in
      let buffer = unsafe {
        self
          .inner
          .buffer
          .load(Ordering::Relaxed, &guard)
          .deref()
          .to_owned()
      };

      self.buffer.replace(buffer);

      atomic::fence(Ordering::Release);

      match self.inner.slot.compare_exchange(
        slot,
        1 << FLAGS_SHIFT,
        Ordering::Acquire,
        Ordering::Relaxed,
      ) {
        Ok(_) => {
          unsafe {
            buffer.write(0, task);
          }
          0
        }
        Err(_) => {
          self.push(task)
        }
      }
    } else {
      let slot = slot >> FLAGS_SHIFT;
      let mut buffer = self.buffer.get();

      // Is the queue full?
      if slot >= buffer.cap as isize {
        // Yes. Grow the underlying buffer.
        unsafe {
          self.resize(2 * buffer.cap);
        }

        buffer = self.buffer.get();
      }

      // Write `task` into the slot.
      unsafe {
        buffer.write(slot, task);
      }

      slot
    }
  }

  /// Take the entire queue via swapping the underlying buffer and converting into a `Vec<T>`
  pub fn take_queue(&self) -> Vec<T> {
    let slot = self.inner.slot.fetch_or(BUFFER_SWAPPED, Ordering::Relaxed);

    // Buffer was previously taken
    if slot & BUFFER_SWAPPED == BUFFER_SWAPPED {
      return vec![];
    }

    let slot = slot >> FLAGS_SHIFT;

    let new = Buffer::alloc(MIN_CAP);

    let guard = &epoch::pin();

    let old = unsafe {
      self
        .inner
        .buffer
        .swap(Owned::new(new).into_shared(guard), Ordering::Release, guard)
        .into_owned()
    };

    unsafe { Vec::from_raw_parts(old.ptr, slot as usize, old.cap) }
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
    let slot = self.inner.slot.fetch_or(BUFFER_SWAPPED, Ordering::Relaxed);

    // Buffer was previously taken
    if slot & BUFFER_SWAPPED == BUFFER_SWAPPED || slot == 0 {
      return vec![];
    }

    let slot = slot >> FLAGS_SHIFT;

    let new = Buffer::alloc(MIN_CAP);

    let guard = &epoch::pin();

    let old = unsafe {
      self
        .inner
        .buffer
        .swap(Owned::new(new).into_shared(guard), Ordering::Release, guard)
        .into_owned()
    };

    unsafe { Vec::from_raw_parts(old.ptr, slot as usize, old.cap) }
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

#[cfg(all(loom, test))]
mod concurrent_tests {
  use super::*;
  use loom::thread;
  use std::convert::identity;

  #[test]
  fn test_concurrent_logic() {
    loom::model(|| {
      let queue = Worker::new();
      let stealer = queue.stealer();

      for i in 0..100 {
        queue.push(i);
      }

      thread::spawn(move || {
        let batch = stealer.take_queue();
        let expected = (0..100).map(identity).collect::<Vec<i32>>();
        assert_eq!(batch, expected);
      })
      .join()
      .unwrap();

      for i in 0..100 {
        queue.push(i);
      }

      let batch = queue.take_queue();
      let expected = (0..100).map(identity).collect::<Vec<i32>>();
      assert_eq!(batch, expected);
    });
  }

  #[test]
  fn test_multi_stealer() {
    loom::model(|| {
      let queue = Worker::new();
      let stealer = queue.stealer();

      let threads: Vec<_> = (0..2)
        .map(|_| {
          let stealer = stealer.clone();
          thread::spawn(move || {
            let batch = stealer.take_queue();
            batch.len()
          })
        })
        .collect();

      for i in 0..50 {
        queue.push(i);
      }

      let received_len: std::thread::Result<usize> =
        threads.into_iter().map(|thread| thread.join()).sum();

      let batch = queue.take_queue();
      assert_eq!(received_len.unwrap() + batch.len(), 50);
    });
  }

  #[test]
  fn test_empty() {
    loom::model(|| {
      let queue: Worker<i32> = Worker::new();
      let stealer = queue.stealer();

      let threads: Vec<_> = (0..2)
        .map(|_| {
          let stealer = stealer.clone();
          thread::spawn(move || {
            let batch = stealer.take_queue();
            batch.len()
          })
        })
        .collect();

      let received_len: std::thread::Result<usize> =
        threads.into_iter().map(|thread| thread.join()).sum();

      assert_eq!(received_len.unwrap(), 0);
    });
  }
}
