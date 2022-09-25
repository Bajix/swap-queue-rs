use super::buffer::Buffer;
use cache_padded::CachePadded;
use std::{
  cell::UnsafeCell,
  mem::{self},
  ptr,
  sync::atomic::AtomicUsize,
};

pub(super) struct Inner<T> {
  pub(super) slot: CachePadded<AtomicUsize>,
  buffer: UnsafeCell<Buffer<T>>,
}

unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Inner<T> {
  pub(super) fn from_slot(slot: CachePadded<AtomicUsize>) -> Self {
    Inner {
      slot,
      buffer: UnsafeCell::new(Buffer::alloc(0)),
    }
  }

  /// Write to offset without checking offset
  pub(super) unsafe fn write_unchecked(&self, offset: isize, task: T) {
    ptr::write((*self.buffer.get()).at(offset), task);
  }

  /// Resize to double previous allocated capacity
  pub(super) unsafe fn resize_doubled(&self) {
    let buffer = &mut *self.buffer.get();
    let new = Buffer::alloc(buffer.cap << 1);

    ptr::copy_nonoverlapping(buffer.at(0), new.at(0), buffer.cap);

    let old = mem::replace(buffer, new);

    // Here we're deallocating the old buffer without dropping individual items as the new buffer is now owner
    old.dealloc();
  }

  pub(crate) unsafe fn to_vec(&self, length: usize) -> Vec<T> {
    let buffer = (*self.buffer.get()).clone();
    buffer.to_vec(length)
  }
}

impl<T> From<T> for Inner<T> {
  fn from(task: T) -> Self {
    let buffer = Buffer::alloc(1);
    unsafe {
      buffer.write(0, task);
    }

    Inner {
      slot: CachePadded::new(AtomicUsize::new(1 << 32)),
      buffer: UnsafeCell::new(buffer),
    }
  }
}
