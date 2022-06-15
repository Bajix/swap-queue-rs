use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tokio::runtime::Builder;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

mod bench_swap_queue {
  use futures::future::join_all;
  use swap_queue::SwapQueue;
  use tokio::{
    runtime::Handle,
    sync::oneshot::{channel, Sender},
  };

  thread_local! {
    static QUEUE: SwapQueue<(u64, Sender<u64>)> = SwapQueue::new();
  }

  async fn push_echo(i: u64) -> u64 {
    {
      let (tx, rx) = channel();

      QUEUE.with(|queue| {
        if let Some(stealer) = queue.push((i, tx)) {
          Handle::current().spawn(async move {
            let batch = stealer.await;

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

  pub async fn bench_batching(batch_size: &u64) {
    let batch: Vec<u64> = join_all((0..*batch_size).map(|i| push_echo(i))).await;

    assert_eq!(batch, (0..*batch_size).collect::<Vec<u64>>())
  }
}

mod bench_swap_queue_v1 {
  use futures::future::join_all;
  use swap_queue_v1::Worker;
  use tokio::{
    runtime::Handle,
    sync::oneshot::{channel, Sender},
  };

  thread_local! {
    static QUEUE: Worker<(u64, Sender<u64>)> = Worker::new();
  }

  async fn push_echo(i: u64) -> u64 {
    {
      let (tx, rx) = channel();

      QUEUE.with(|queue| {
        if let Some(stealer) = queue.push((i, tx)) {
          Handle::current().spawn(async move {
            let batch = stealer.take().await;

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

  pub async fn bench_batching(batch_size: &u64) {
    let batch: Vec<u64> = join_all((0..*batch_size).map(|i| push_echo(i))).await;

    assert_eq!(batch, (0..*batch_size).collect::<Vec<u64>>())
  }
}

mod bench_crossbeam {
  use crossbeam_deque::{Steal, Worker};
  use futures::future::join_all;
  use tokio::{
    runtime::Handle,
    sync::oneshot::{channel, Sender},
  };

  thread_local! {
    static QUEUE: Worker<(u64, Sender<u64>)> = Worker::new_fifo();
  }

  async fn push_echo(i: u64) -> u64 {
    let (tx, rx) = channel();

    QUEUE.with(|queue| {
      // crossbeam_deque::Worker could be patched to return slot written, so we're going to give this the benefit of that potential optimization
      if i.eq(&0) {
        let stealer = queue.stealer();

        Handle::current().spawn(async move {
          let batch: Vec<(u64, Sender<u64>)> = std::iter::from_fn(|| loop {
            match stealer.steal() {
              Steal::Success(task) => break Some(task),
              Steal::Retry => continue,
              Steal::Empty => break None,
            }
          })
          .collect();

          batch.into_iter().for_each(|(i, tx)| {
            tx.send(i).ok();
          });
        });
      }

      queue.push((i, tx));
    });

    rx.await.unwrap()
  }

  pub async fn bench_batching(batch_size: &u64) {
    let batch: Vec<u64> = join_all((0..*batch_size).map(|i| push_echo(i))).await;

    assert_eq!(batch, (0..*batch_size).collect::<Vec<u64>>())
  }
}

mod bench_tokio {
  use futures::future::join_all;
  use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot},
  };

  fn make_reactor() -> mpsc::UnboundedSender<(u64, oneshot::Sender<u64>)> {
    let (tx, mut rx) = mpsc::unbounded_channel();

    Handle::current().spawn(async move {
      loop {
        if let Some(task) = rx.recv().await {
          let batch: Vec<(u64, oneshot::Sender<u64>)> = std::iter::once(task)
            .chain(std::iter::from_fn(|| rx.try_recv().ok()))
            .collect();

          batch.into_iter().for_each(|(i, tx)| {
            tx.send(i).ok();
          });
        }
      }
    });

    tx
  }

  async fn push_echo(i: u64) -> u64 {
    thread_local! {
      static QUEUE: mpsc::UnboundedSender<(u64, oneshot::Sender<u64>)> = make_reactor();
    }

    let (tx, rx) = oneshot::channel();

    QUEUE.with(|queue_tx| {
      queue_tx.send((i, tx)).ok();
    });

    rx.await.unwrap()
  }

  pub async fn bench_batching(batch_size: &u64) {
    let batch: Vec<u64> = join_all((0..*batch_size).map(|i| push_echo(i))).await;

    assert_eq!(batch, (0..*batch_size).collect::<Vec<u64>>())
  }
}

mod bench_flume {
  use flume::{self, Sender};
  use futures::future::join_all;
  use tokio::{runtime::Handle, sync::oneshot};

  fn make_reactor() -> Sender<(u64, oneshot::Sender<u64>)> {
    let (tx, rx) = flume::unbounded();

    Handle::current().spawn(async move {
      loop {
        if let Some(task) = rx.recv_async().await.ok() {
          let batch: Vec<(u64, oneshot::Sender<u64>)> = std::iter::once(task)
            .chain(std::iter::from_fn(|| rx.try_recv().ok()))
            .collect();

          batch.into_iter().for_each(|(i, tx)| {
            tx.send(i).ok();
          });
        }
      }
    });

    tx
  }

  async fn push_echo(i: u64) -> u64 {
    thread_local! {
      static QUEUE: Sender<(u64, oneshot::Sender<u64>)> = make_reactor();
    }

    let (tx, rx) = oneshot::channel();

    QUEUE.with(|queue_tx| {
      queue_tx.send((i, tx)).ok();
    });

    rx.await.unwrap()
  }

  pub async fn bench_batching(batch_size: &u64) {
    let batch: Vec<u64> = join_all((0..*batch_size).map(|i| push_echo(i))).await;

    assert_eq!(batch, (0..*batch_size).collect::<Vec<u64>>())
  }
}

fn criterion_benchmark(c: &mut Criterion) {
  let rt = Builder::new_current_thread().build().unwrap();

  let mut push_tests = c.benchmark_group("Push");
  push_tests.warm_up_time(Duration::from_millis(10));
  push_tests.measurement_time(Duration::from_secs(1));
  push_tests.sample_size(50);

  for n in 0..=12 {
    let batch_size: u64 = 1 << n;
    push_tests.bench_with_input(
      BenchmarkId::new("swap-queue", batch_size),
      &batch_size,
      |b, batch_size| {
        b.iter_batched(
          || swap_queue::SwapQueue::new(),
          |queue| {
            for i in 0..*batch_size {
              queue.push(i);
            }
          },
          BatchSize::PerIteration,
        )
      },
    );

    push_tests.bench_with_input(
      BenchmarkId::new("swap-queue-v1.0.0", batch_size),
      &batch_size,
      |b, batch_size| {
        b.iter_batched(
          || swap_queue_v1::Worker::new(),
          |queue| {
            for i in 0..*batch_size {
              queue.push(i);
            }
          },
          BatchSize::PerIteration,
        )
      },
    );

    push_tests.bench_with_input(
      BenchmarkId::new("crossbeam", batch_size),
      &batch_size,
      |b, batch_size| {
        b.iter_batched(
          || crossbeam_deque::Worker::new_fifo(),
          |queue| {
            for i in 0..*batch_size {
              queue.push(i);
            }
          },
          BatchSize::PerIteration,
        )
      },
    );

    push_tests.bench_with_input(
      BenchmarkId::new("flume", batch_size),
      &batch_size,
      |b, batch_size| {
        b.iter_batched(
          || flume::unbounded(),
          |(tx, _rx)| {
            for i in 0..*batch_size {
              tx.send(i).ok();
            }
          },
          BatchSize::PerIteration,
        )
      },
    );

    push_tests.bench_with_input(
      BenchmarkId::new("tokio::mpsc", batch_size),
      &batch_size,
      |b, batch_size| {
        b.iter_batched(
          || tokio::sync::mpsc::unbounded_channel(),
          |(tx, _rx)| {
            for i in 0..*batch_size {
              tx.send(i).ok();
            }
          },
          BatchSize::PerIteration,
        )
      },
    );
  }

  push_tests.finish();

  let mut take_tests = c.benchmark_group("Take");
  take_tests.warm_up_time(Duration::from_millis(10));
  take_tests.measurement_time(Duration::from_secs(1));
  take_tests.sample_size(50);

  for n in 0..=12 {
    let batch_size: u64 = 1 << n;
    take_tests.bench_with_input(
      BenchmarkId::new("swap-queue", batch_size),
      &batch_size,
      |b, batch_size| {
        b.to_async(&rt).iter_batched(
          || {
            let worker = swap_queue::SwapQueue::new();
            let stealer = worker.push(0).unwrap();
            for i in 1..*batch_size {
              worker.push(i);
            }

            stealer
          },
          |stealer| async move { stealer.await },
          BatchSize::PerIteration,
        );
      },
    );

    take_tests.bench_with_input(
      BenchmarkId::new("swap-queue-v1.0.0", batch_size),
      &batch_size,
      |b, batch_size| {
        b.to_async(&rt).iter_batched(
          || {
            let worker = swap_queue_v1::Worker::new();
            let stealer = worker.push(0).unwrap();
            for i in 1..*batch_size {
              worker.push(i);
            }

            stealer
          },
          |stealer| async move { stealer.take().await },
          BatchSize::PerIteration,
        );
      },
    );

    take_tests.bench_with_input(
      BenchmarkId::new("crossbeam", batch_size),
      &batch_size,
      |b, batch_size| {
        b.iter_batched(
          || {
            let worker = crossbeam_deque::Worker::new_fifo();
            let stealer = worker.stealer();
            for i in 1..*batch_size {
              worker.push(i);
            }

            stealer
          },
          |stealer| {
            let _: Vec<u64> = std::iter::from_fn(|| loop {
              match stealer.steal() {
                crossbeam_deque::Steal::Success(task) => break Some(task),
                crossbeam_deque::Steal::Retry => continue,
                crossbeam_deque::Steal::Empty => break None,
              }
            })
            .collect();
          },
          BatchSize::PerIteration,
        );
      },
    );

    take_tests.bench_with_input(
      BenchmarkId::new("flume", batch_size),
      &batch_size,
      |b, batch_size| {
        b.iter_batched(
          || {
            let (tx, rx) = flume::unbounded();
            for i in 1..*batch_size {
              tx.send(i).ok();
            }
            rx
          },
          |rx| {
            let _: Vec<u64> = rx.try_iter().collect();
          },
          BatchSize::PerIteration,
        );
      },
    );

    take_tests.bench_with_input(
      BenchmarkId::new("tokio::mpsc", batch_size),
      &batch_size,
      |b, batch_size| {
        b.iter_batched(
          || {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            for i in 1..*batch_size {
              tx.send(i).ok();
            }
            rx
          },
          |mut rx| {
            let _: Vec<u64> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
          },
          BatchSize::PerIteration,
        );
      },
    );
  }

  take_tests.finish();

  let mut async_batching_tests = c.benchmark_group("Batching");
  async_batching_tests.warm_up_time(Duration::from_millis(10));
  async_batching_tests.measurement_time(Duration::from_secs(1));
  async_batching_tests.sample_size(50);

  for n in 0..=12 {
    let batch_size: u64 = 1 << n;

    async_batching_tests.bench_with_input(
      BenchmarkId::new("swap-queue", batch_size),
      &batch_size,
      |b, batch_size| {
        b.to_async(&rt)
          .iter(|| bench_swap_queue::bench_batching(batch_size))
      },
    );

    async_batching_tests.bench_with_input(
      BenchmarkId::new("swap-queue-v1.0.0", batch_size),
      &batch_size,
      |b, batch_size| {
        b.to_async(&rt)
          .iter(|| bench_swap_queue_v1::bench_batching(batch_size))
      },
    );

    async_batching_tests.bench_with_input(
      BenchmarkId::new("crossbeam", batch_size),
      &batch_size,
      |b, batch_size| {
        b.to_async(&rt)
          .iter(|| bench_crossbeam::bench_batching(batch_size))
      },
    );

    async_batching_tests.bench_with_input(
      BenchmarkId::new("flume", batch_size),
      &batch_size,
      |b, batch_size| {
        b.to_async(&rt)
          .iter(|| bench_flume::bench_batching(batch_size))
      },
    );

    async_batching_tests.bench_with_input(
      BenchmarkId::new("tokio::mpsc", batch_size),
      &batch_size,
      |b, batch_size| {
        b.to_async(&rt)
          .iter(|| bench_tokio::bench_batching(batch_size))
      },
    );
  }

  async_batching_tests.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
