//!
//! A lock-free thread-owned queue whereby tasks are taken by stealers in entirety via buffer swapping. This is meant to be used [`thread_local`] paired with [`tokio::task::spawn`] as a constant-time take-all batching mechanism that outperforms [`crossbeam_deque::Worker`](https://docs.rs/crossbeam-deque/0.8.1/crossbeam_deque/struct.Worker.html), and [`tokio::sync::mpsc`] for batching.
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

mod swap_queue;
pub use swap_queue::{Stealer, SwapQueue};
