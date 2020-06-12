#! /bin/sh
export RUSTFLAGS="--emit=asm"
[ -n "$BERBALANG_LOG" ] || BERBALANG_LOG=info
export BERBALANG_LOG
export RUST_BACKTRACE=1
cargo run --release
