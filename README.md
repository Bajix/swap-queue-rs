# Swap Queue

![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Cargo](https://img.shields.io/crates/v/swap-queue.svg)](https://crates.io/crates/swap-queue)
[![Documentation](https://docs.rs/swap-queue/badge.svg)](https://docs.rs/swap-queue)
[![CI](https://github.com/Bajix/swap-queue-rs/actions/workflows/run-tests.yml/badge.svg)](https://github.com/Bajix/swap-queue-rs/actions/workflows/run-tests.yml)

A lock-free thread-owned queue whereby tasks are taken by stealers in entirety via buffer swapping. For batching use-cases, this has the advantage that all tasks can be taken as a single batch in constant time irregardless of batch size, whereas alternatives using [`crossbeam_deque::Worker`](https://docs.rs/crossbeam-deque/0.8.1/crossbeam_deque/struct.Worker.html) and [`tokio::sync::mpsc`](https://docs.rs/tokio/1.14.0/tokio/sync/mpsc/index.html) need to collect each task separately and situationally lack a clear cutoff point. This design ensures that should you be waiting on a resource such as a connection to be available, that once it is so there is no further delay before a task batch can be processed. While push behavior alone is slower than [`crossbeam_deque::Worker`](https://docs.rs/crossbeam-deque/0.8.1/crossbeam_deque/struct.Worker.html) and faster than [`tokio::sync::mpsc`](https://docs.rs/tokio/1.14.0/tokio/sync/mpsc/index.html), overall batching performance is around ~11-19% faster than [`crossbeam_deque::Worker`](https://docs.rs/crossbeam-deque/0.8.1/crossbeam_deque/struct.Worker.html), and ~28-45% faster than [`tokio::sync::mpsc`](https://docs.rs/tokio/1.14.0/tokio/sync/mpsc/index.html) on ARM and there is never a slow cutoff between batches.

## Example

```rust
use swap_queue::SwapQueue;
use tokio::{
  runtime::Handle,
  sync::oneshot::{channel, Sender},
};

// Jemalloc makes this library substantially faster
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// SwapQueue needs to be thread local because it is !Sync
thread_local! {
  static QUEUE: SwapQueue<(u64, Sender<u64>)> = SwapQueue::new();
}

// This mechanism will batch optimally without overhead within an async-context because spawn will happen after things already scheduled
async fn push_echo(i: u64) -> u64 {
  {
    let (tx, rx) = channel();

    QUEUE.with(|queue| {
      // A new stealer is returned whenever the buffer is new or was empty
      if let Some(stealer) = queue.push((i, tx)) {
        Handle::current().spawn(async move {
          // Take the underlying buffer in entirety; the next push will return a new Stealer
          let batch = stealer.await;

          // Some sort of batched operation, such as a database query

          batch.into_iter().for_each(|(i, tx)| {
            tx.send(i).ok();
          });
        });
      }
    });

    rx
  }
  .await
  .unwrap()
}
```

## Benchmarks

Benchmarks ran on t4g.medium using ami-06391d741144b83c2

### Async Batching

<img src="https://raw.githubusercontent.com/Bajix/swap-queue-benchmarks/master/Batching/64/report/violin.svg" alt="Benchmarks, 64 tasks" width="100%"/>
<img src="https://raw.githubusercontent.com/Bajix/swap-queue-benchmarks/master/Batching/128/report/violin.svg" alt="Benchmarks, 128 tasks" width="100%"/>
<img src="https://raw.githubusercontent.com/Bajix/swap-queue-benchmarks/master/Batching/256/report/violin.svg" alt="Benchmarks, 256 tasks" width="100%"/>
<img src="https://raw.githubusercontent.com/Bajix/swap-queue-benchmarks/master/Batching/512/report/violin.svg" alt="Benchmarks, 512 tasks" width="100%"/>
<img src="https://raw.githubusercontent.com/Bajix/swap-queue-benchmarks/master/Batching/1024/report/violin.svg" alt="Benchmarks, 1024 tasks" width="100%"/>

### Push

<img src="https://raw.githubusercontent.com/Bajix/swap-queue-benchmarks/master/Push/report/lines.svg" alt="Benchmarks, 1024 tasks" width="100%"/>

### Batch collecting

<img src="https://raw.githubusercontent.com/Bajix/swap-queue-benchmarks/master/Take/report/lines.svg" alt="Benchmarks, 1024 tasks" width="100%"/>

CI tested under ThreadSanitizer, LeakSanitizer and Miri.
