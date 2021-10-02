# Swap Queue

![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Cargo](https://img.shields.io/crates/v/swap-queue.svg)](https://crates.io/crates/swap-queue)
[![Documentation](https://docs.rs/swap-queue/badge.svg)](https://docs.rs/swap-queue)

A lockfree thread-owned queue whereby tasks are taken by threadsafe stealers in entirety via buffer swapping instead of task popping
