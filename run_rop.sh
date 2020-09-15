#! /usr/bin/env bash

BERBALANG_LOG=info,berbalib::emulator::hatchery=trace cargo run $CARGOFLAGS --bin run_rop --features disassemble_trace -- $*
