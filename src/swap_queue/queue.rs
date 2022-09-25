use super::{buffer::MIN_CAP, inner::Inner, stealer::Stealer};
use std::{
  cell::RefCell,
  sync::{atomic::Ordering, Arc, Weak},
};

// Indicate that buffer has been taken
pub(super) const BUFFER_TAKEN: usize = 1 << 0;

// Designates that write is in progress
pub(super) const WRITE_IN_PROGRESS: usize = 1 << 31;

pub struct SwapQueue<T: Send + Sized> {
  inner: RefCell<Weak<Inner<T>>>,
}

impl<T: Send + Sized> SwapQueue<T> {
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
    SwapQueue {
      inner: RefCell::new(Weak::new()),
    }
  }

  fn inner(&self) -> Option<Arc<Inner<T>>> {
    self.inner.borrow().upgrade()
  }

  /// Write to the next slot, returning a Stealer at the start of a new batch
  pub fn push(&self, task: T) -> Option<Stealer<T>> {
    if let Some(inner) = self.inner() {
      let slot = inner.slot.fetch_add(WRITE_IN_PROGRESS, Ordering::Relaxed);

      if (slot & BUFFER_TAKEN).eq(&BUFFER_TAKEN) {
        let inner: Arc<Inner<T>> = Arc::new(task.into());
        self.inner.replace(Arc::downgrade(&inner));
        Some(inner.into())
      } else {
        let index = slot >> 32;
        if (index & (index - 1)).eq(&0) && index >= MIN_CAP {
          unsafe {
            inner.resize_doubled();
          }
        }

        unsafe {
          inner.write_unchecked(index as isize, task);
        }

        inner.slot.fetch_add(WRITE_IN_PROGRESS, Ordering::Relaxed);

        None
      }
    } else {
      let inner: Arc<Inner<T>> = Arc::new(task.into());
      self.inner.replace(Arc::downgrade(&inner));
      Some(inner.into())
    }
  }
}

impl<T> Default for SwapQueue<T>
where
  T: Send + Sized,
{
  fn default() -> Self {
    Self::new()
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
  fn it_resizes() {
    model!({
      let queue = SwapQueue::new();
      let stealer = queue.push(0).unwrap();

      for i in 1..128 {
        queue.push(i);
      }

      let batch: Vec<i32> = stealer.into();
      let expected = (0..128).collect::<Vec<i32>>();

      assert_eq!(batch, expected);
    });
  }

  #[test]
  fn it_makes_new_stealer_per_batch() {
    model!({
      let queue = SwapQueue::new();
      let stealer = queue.push(0).unwrap();

      queue.push(1);
      queue.push(2);

      let data: Vec<i32> = stealer.into();

      assert_eq!(data, vec![0, 1, 2]);

      let stealer = queue.push(3).unwrap();
      queue.push(4);
      queue.push(5);

      let data: Vec<i32> = stealer.into();

      assert_eq!(data, vec![3, 4, 5]);
    });
  }

  #[cfg(not(loom))]
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

  #[test]
  fn stealer_takes_blocking() {
    model!({
      let queue = SwapQueue::new();
      let stealer = queue.push(0).unwrap();

      for i in 1..128 {
        queue.push(i);
      }

      thread::spawn(move || {
        let _: Vec<i32> = stealer.into();
      })
      .join()
      .unwrap();
    });
  }

  #[cfg(not(loom))]
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

  #[cfg(loom)]
  #[tokio::test]
  async fn queue_drops() {
    loom::model(|| {
      let queue = SwapQueue::new();
      let stealer = queue.push(0).unwrap();

      for i in 1..128 {
        queue.push(i);
      }

      drop(queue);

      let batch: Vec<i32> = stealer.into();
      let expected = (0..128).collect::<Vec<i32>>();

      assert_eq!(batch, expected);
    });
  }
}
