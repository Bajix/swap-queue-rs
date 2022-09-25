use std::{mem, ptr};

/// A buffer that holds tasks in a worker queue.
///
/// This is just a pointer to the buffer and its length - dropping an instance of this struct will
/// *not* deallocate the buffer.
///

// Minimum buffer capacity.
pub(super) const MIN_CAP: usize = 64;

pub(super) struct Buffer<T> {
  /// Pointer to the allocated memory.
  ptr: *mut T,

  /// Capacity of the buffer. Always a power of two.
  pub(super) cap: usize,
}

unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Send> Sync for Buffer<T> {}

impl<T> Buffer<T> {
  /// Allocates a new buffer with the specified capacity.
  pub(super) fn alloc(cap: usize) -> Buffer<T> {
    let mut v = Vec::with_capacity(cap);
    let ptr = v.as_mut_ptr();
    mem::forget(v);

    Buffer { ptr, cap }
  }

  /// Deallocates the buffer.
  pub(super) unsafe fn dealloc(self) {
    drop(Vec::from_raw_parts(self.ptr, 0, self.cap));
  }

  /// Returns a pointer to the task at the specified `index`.
  #[inline]
  pub(super) unsafe fn at(&self, index: isize) -> *mut T {
    self.ptr.offset(index)
  }

  /// Writes `task` into the specified `index`.
  pub(super) unsafe fn write(&self, index: isize, task: T) {
    ptr::write_volatile(self.at(index), task)
  }

  pub(super) unsafe fn to_vec(self, length: usize) -> Vec<T> {
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
