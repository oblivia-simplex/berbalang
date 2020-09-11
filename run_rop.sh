#! /usr/bin/env bash

BERBALANG_LOG=info,berbalib::emulator::hatchery=trace cargo run --bin run_rop --features disassemble_trace -- $*
