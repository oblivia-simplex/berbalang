#! /bin/sh
[ -n "$BERBALANG_LOG" ] || BERBALANG_LOG=info
export BERBALANG_LOG

if [ -f "/usr/local/bin/berbalang" ]; then
  # only makes sense inside of docker
  /usr/local/bin/berbalang $*
else
  export RUST_BACKTRACE=1
  export RUSTFLAGS="--emit=asm"
  cargo run --release $*
fi
