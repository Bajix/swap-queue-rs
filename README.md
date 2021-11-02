# Swap Queue

![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Cargo](https://img.shields.io/crates/v/swap-queue.svg)](https://crates.io/crates/swap-queue)
[![Documentation](https://docs.rs/swap-queue/badge.svg)](https://docs.rs/swap-queue)
[![CI](https://github.com/Bajix/swap-queue-rs/actions/workflows/run-tests.yml/badge.svg)](https://github.com/Bajix/swap-queue-rs/actions/workflows/run-tests.yml)

A lock-free thread-owned queue whereby tasks are taken by stealers in entirety via buffer swapping. This is meant to be used [`thread_local`] paired with [`tokio::task::spawn`] as a highly-performant take-all batching mechanism and is around ~11-19% faster than [`crossbeam::deque::Worker`], and ~28-45% faster than [`tokio::sync::mpsc`] on ARM.

## Example

```rust
use swap_queue::Worker;
use tokio::{
  runtime::Handle,
  sync::oneshot::{channel, Sender},
};

// Jemalloc makes this library substantially faster
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// Worker needs to be thread local because it is !Sync
thread_local! {
  static QUEUE: Worker<(u64, Sender<u64>)> = Worker::new();
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
          let batch = stealer.take().await;

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

Benchmarks ran on t4g.medium running Amazon Linux 2 AMI (HVM)

<img src="target/criterion/Batching/64/report/violin.svg" alt="Benchmarks, 64 tasks" width="100%"/>
<img src="target/criterion/Batching/128/report/violin.svg" alt="Benchmarks, 128 tasks" width="100%"/>
<img src="target/criterion/Batching/256/report/violin.svg" alt="Benchmarks, 256 tasks" width="100%"/>
<img src="target/criterion/Batching/512/report/violin.svg" alt="Benchmarks, 512 tasks" width="100%"/>
<img src="target/criterion/Batching/1024/report/violin.svg" alt="Benchmarks, 1024 tasks" width="100%"/>

CI tested under ThreadSanitizer, LeakSanitizer, Miri and Loom.
