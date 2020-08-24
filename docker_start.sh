#! /usr/bin/env bash


LOGS=${HOME}/logs
mkdir -p "$LOGS"

GADGETS=$PWD/gadgets
mkdir -p "$GADGETS"

BINARIES=$PWD/binaries
mkdir -p "$BINARIES"

EXPERIMENTS="$1"

NUMBER_OF_TRIALS="$2"

if [ -z "$NUMBER_OF_TRIALS" ]; then
  echo "Usage: $0 <experiment specification directory> <number of trials>"
  exit 1
fi

if [ -n "$REBUILD" ]; then
  docker build -t berbalang .
fi

docker container run \
  --mount src="$LOGS",dst=/root/logs,type=bind \
  --mount src="$GADGETS",dst=/root/gadgets,type=bind,readonly \
  --mount src="$BINARIES",dst=/root/binaries,type=bind,readonly \
  --mount src="$EXPERIMENTS",dst=/root/experiments,type=bind,readonly \
  berbalang:latest \
  /root/trials.sh /root/experiments $NUMBER_OF_TRIALS /root/logs
