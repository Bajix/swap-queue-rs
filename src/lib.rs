use crossbeam::{
  epoch::{self, Atomic, Owned, Shared},
  utils::CachePadded,
};
use oneshot::{channel, Receiver, Sender};
use std::{cell::Cell, fmt, marker::PhantomData, mem, ptr, sync::Arc};

#[cfg(loom)]
use loom::sync::atomic::{AtomicIsize, Ordering};

#[cfg(not(loom))]
use std::sync::atomic::{AtomicIsize, Ordering};

// Current buffer index
const BUFFER_IDX: isize = 1 << 0;

// Designates that write is in progress
const WRITE_IN_PROGRESS: isize = 1 << 1;

// Designates how many bits are set aside for flags
const FLAGS_SHIFT: isize = 1;

// Slot increments both for reads and writes, therefore we shift slot an extra bit to extract length
const LENGTH_SHIFT: isize = FLAGS_SHIFT + 1;

// Minimum buffer capacity.
const MIN_CAP: usize = 64;

/// A buffer that holds tasks in a worker queue.
///
/// This is just a pointer to the buffer and its length - dropping an instance of this struct will
/// *not* deallocate the buffer.
struct Buffer<T> {
  /// Slot that represents the index offset and buffer idx
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

struct Inner<T> {
  slot: AtomicIsize,
  buffers: (
    CachePadded<Atomic<Buffer<T>>>,
    CachePadded<Atomic<Buffer<T>>>,
  ),
}

impl<T> Inner<T> {
  fn get_buffer(&self, slot: isize) -> &CachePadded<Atomic<Buffer<T>>> {
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
          let length = (slot >> LENGTH_SHIFT) - (buffer.slot >> LENGTH_SHIFT);

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
      slot: AtomicIsize::new(0),
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
  unsafe fn resize(&self, buffer: &mut Buffer<T>, slot: isize) {
    let length = (slot >> LENGTH_SHIFT) - (buffer.slot >> LENGTH_SHIFT);

    // Allocate a new buffer and copy data from the old buffer to the new one.
    let new = Buffer::alloc(buffer.slot, buffer.cap * 2);

    ptr::copy_nonoverlapping(buffer.at(0), new.at(0), length as usize);

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
      let index = (slot >> LENGTH_SHIFT) - (buffer.slot >> LENGTH_SHIFT);

      // Is the queue full?
      if index >= buffer.cap as isize {
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
      tx.send(unsafe { buffer.to_vec(index as usize) }).ok();

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
        buffer.to_vec(((slot >> LENGTH_SHIFT) - (buffer.slot >> LENGTH_SHIFT)) as usize)
      }
    }
  }

  /// Take the entire queue via swapping the underlying buffer and converting into a `Vec<T>`, blocking only if write in progress
  pub fn take_blocking(self) -> Vec<T> {
    let Stealer { rx, inner } = self;

    let slot = inner.slot.fetch_xor(BUFFER_IDX, Ordering::Relaxed);

    // Worker will see the buffer has swapped when confirming length increment
    if slot & WRITE_IN_PROGRESS == WRITE_IN_PROGRESS {
      // Writer can never be dropped mid-write, therefore RecvError cannot occur
      rx.recv().unwrap()
    } else {
      let guard = &epoch::pin();

      let buffer = inner
        .get_buffer(slot)
        .swap(Shared::null(), Ordering::Acquire, guard);

      unsafe {
        let buffer = *buffer.into_owned();
        buffer.to_vec(((slot >> LENGTH_SHIFT) - (buffer.slot >> LENGTH_SHIFT)) as usize)
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
  fn it_resizes() {
    model!({
      let queue = Worker::new();
      let stealer = queue.push(0).unwrap();

      for i in 1..256 {
        queue.push(i);
      }

      let batch = stealer.take_blocking();
      let expected = (0..256).collect::<Vec<i32>>();

      assert_eq!(batch, expected);
    });
  }

  #[test]
  fn stealer_takes() {
    model!({
      let queue = Worker::new();
      let stealer = queue.push(0).unwrap();

      for i in 1..100 {
        queue.push(i);
      }

      thread::spawn(move || {
        stealer.take_blocking();
      })
      .join()
      .unwrap();
    });
  }

  #[test]
  fn takes_while_pushing() {
    model!({
      let (tx, rx) = channel::<Stealer<i32>>();

      let handles = vec![
        thread::spawn(move || {
          let queue = Worker::new();
          let stealer = queue.push(0).unwrap();

          for i in 1..64 {
            queue.push(i);
          }

          tx.send(stealer).unwrap();

          for i in 0..64 {
            queue.push(i);
          }
        }),
        thread::spawn(move || {
          let stealer = rx.recv().unwrap();
          stealer.take_blocking();
        }),
      ];

      for handle in handles {
        handle.join().unwrap();
      }
    });
  }
}
