#! /bin/sh
export RUSTFLAGS="--emit=asm"
RUST_BACKTRACE=1 cargo run --release
