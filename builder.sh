#! /bin/sh

nix-shell --command "rustup default stable && cargo build --release --bin berbalang"

