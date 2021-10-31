# Swap Queue

![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Cargo](https://img.shields.io/crates/v/swap-queue.svg)](https://crates.io/crates/swap-queue)
[![Documentation](https://docs.rs/swap-queue/badge.svg)](https://docs.rs/swap-queue)

A lock-free thread-owned queue whereby tasks are taken by stealers in entirety via buffer swapping. This is meant to be used [`thread_local`] paired with [`tokio::task::spawn`] as a highly-performant take-all batching mechanism and is around ~11-19% faster than [`crossbeam::deque::Worker`], and ~28-45% faster than [`tokio::sync::mpsc`] on ARM.

<img src="target/criterion/Batching/64/report/violin.svg" alt="Benchmarks, 64 tasks" width="100%"/>
<img src="target/criterion/Batching/128/report/violin.svg" alt="Benchmarks, 128 tasks" width="100%"/>
<img src="target/criterion/Batching/256/report/violin.svg" alt="Benchmarks, 256 tasks" width="100%"/>
<img src="target/criterion/Batching/512/report/violin.svg" alt="Benchmarks, 512 tasks" width="100%"/>
<img src="target/criterion/Batching/1024/report/violin.svg" alt="Benchmarks, 1024 tasks" width="100%"/>

Benchmarks ran on t4g.medium running Amazon Linux 2 AMI (HVM)
