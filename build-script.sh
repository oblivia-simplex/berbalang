#!/usr/bin/env bash

set -e

build_dependency () {
  download_url="${1}"
  unpack_cmd="${2}"
  download_target="${3}"
  dir_name="${4}"

  wget "${download_url}" -O "${download_target}" && ${unpack_cmd} "${download_target}"
  pushd "${dir_name}"
  make && make install
  popd
  rm -rf "${dir_name}"
}

echo "Sleeping for 5s to let networking settle"
sleep 5

echo "Updating the cache"
apt update -y
echo "Installing tooling"
apt install -y build-essential clang llvm-dev python unzip wget

mkdir -p /tmp/work
pushd /tmp/work

echo "Building unicorn emulator library"
build_dependency https://github.com/unicorn-engine/unicorn/archive/1.0.1.zip "unzip" unicorn_src.zip unicorn-1.0.1

echo "Building capstone disassembly library"
build_dependency https://github.com/aquynh/capstone/archive/4.0.2.tar.gz "tar -xvzf" capstone_src.tar.gz capstone-4.0.2
