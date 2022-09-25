use super::{
  inner::Inner,
  queue::{BUFFER_TAKEN, WRITE_IN_PROGRESS},
};
use std::{
  future::Future,
  hint,
  pin::Pin,
  sync::{
    atomic::{fence, Ordering},
    Arc,
  },
  task::{Context, Poll},
};

enum State {
  Unbounded,
  WriteInProgress { length: usize },
  BufferTaken,
}

/// Stealers take ownership of buffers from the underlying queue
pub struct Stealer<T> {
  state: State,
  pub(super) inner: Arc<Inner<T>>,
}

impl<T> Stealer<T> {
  unsafe fn take_buffer(&mut self, length: usize) -> Vec<T> {
    self.state = State::BufferTaken;
    self.inner.to_vec(length)
  }
}

impl<T> From<Arc<Inner<T>>> for Stealer<T> {
  fn from(inner: Arc<Inner<T>>) -> Self {
    Stealer {
      state: State::Unbounded,
      inner,
    }
  }
}

impl<T> Future for Stealer<T> {
  type Output = Vec<T>;

  fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
    match self.state {
      State::Unbounded => {
        let slot = self.inner.slot.fetch_or(BUFFER_TAKEN, Ordering::Relaxed);

        if (slot & WRITE_IN_PROGRESS).eq(&WRITE_IN_PROGRESS) {
          cx.waker().wake_by_ref();
          unsafe { self.get_unchecked_mut().state = State::WriteInProgress { length: slot >> 32 } }
          Poll::Pending
        } else {
          let buffer = unsafe { self.get_unchecked_mut().take_buffer(slot >> 32) };

          Poll::Ready(buffer)
        }
      }
      State::WriteInProgress { length } => {
        let slot = self.inner.slot.load(Ordering::Relaxed);

        if (slot & WRITE_IN_PROGRESS).eq(&WRITE_IN_PROGRESS) {
          cx.waker().wake_by_ref();
          Poll::Pending
        } else {
          let buffer = unsafe { self.get_unchecked_mut().take_buffer(length) };

          Poll::Ready(buffer)
        }
      }
      State::BufferTaken => unreachable!(),
    }
  }
}

impl<T> Into<Vec<T>> for Stealer<T> {
  fn into(mut self) -> Vec<T> {
    let slot = self.inner.slot.fetch_or(BUFFER_TAKEN, Ordering::Relaxed);

    if (slot & WRITE_IN_PROGRESS).eq(&WRITE_IN_PROGRESS) {
      loop {
        hint::spin_loop();

        if (self.inner.slot.load(Ordering::Relaxed) & WRITE_IN_PROGRESS).eq(&WRITE_IN_PROGRESS) {
          fence(Ordering::Acquire);
        } else {
          break;
        }
      }
    }

    unsafe { self.take_buffer(slot >> 32) }
  }
}

impl<T> Drop for Stealer<T> {
  fn drop(&mut self) {
    match self.state {
      State::Unbounded => {
        let slot = self.inner.slot.fetch_or(BUFFER_TAKEN, Ordering::Relaxed);

        if (slot & WRITE_IN_PROGRESS).eq(&WRITE_IN_PROGRESS) {
          loop {
            hint::spin_loop();

            if (self.inner.slot.load(Ordering::Relaxed) & WRITE_IN_PROGRESS).eq(&WRITE_IN_PROGRESS)
            {
              fence(Ordering::Acquire);
            } else {
              break;
            }
          }
        }

        unsafe {
          self.inner.to_vec(slot >> 32);
        }
      }
      State::WriteInProgress { length } => unsafe {
        self.inner.to_vec(length);
      },
      State::BufferTaken => {}
    }
  }
}
