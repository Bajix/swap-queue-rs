RUST_BACKTRACE=full RUSTFLAGS="--cfg loom -Z sanitizer=thread" cargo test -Z build-std --target x86_64-apple-darwin --release
