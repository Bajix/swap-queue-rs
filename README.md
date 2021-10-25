# Swap Queue

![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Cargo](https://img.shields.io/crates/v/swap-queue.svg)](https://crates.io/crates/swap-queue)
[![Documentation](https://docs.rs/swap-queue/badge.svg)](https://docs.rs/swap-queue)

A lock-free thread-owned queue whereby tasks are taken by stealers in entirety via buffer swapping. This is meant to be used [`thread_local`] paired with [`tokio::task::spawn`] as a performant take-all batching mechanism.
