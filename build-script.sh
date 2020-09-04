#!/usr/bin/env bash

err () {
  echo "Error at line $LINENO"
}

trap err ERR

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
echo "################################"
echo "# Building berbalang started!  #"
echo "################################"

echo "Sleeping for 5s to let networking settle"
sleep 5

echo "Updating the cache"
apt update -y
echo "Installing tooling"
apt install -y build-essential clang llvm-dev python unzip wget curl git

mkdir -p /tmp/work
pushd /tmp/work

echo "Building unicorn emulator library"
build_dependency https://github.com/unicorn-engine/unicorn/archive/1.0.1.zip "unzip" unicorn_src.zip unicorn-1.0.1

echo "Building capstone disassembly library"
build_dependency https://github.com/aquynh/capstone/archive/4.0.2.tar.gz "tar -xvzf" capstone_src.tar.gz capstone-4.0.2

echo "Installing rustup"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y
rm -f ./rustup-init.sh
source "${HOME}/.cargo/env"

echo "Building berbalang"
git clone https://github.com/oblivia-simplex/berbalang
pushd berbalang
cargo build --release
find target/release -maxdepth 2 -type f -executable -exec strip -s {} +
cp -v target/release/berbalang /root/berbalang

echo "Creating tarball to be pulled by lxc"
pushd /
tar cvJf berbalang.tar.xz /root/berbalang /usr/lib/libunicorn*

echo "################################"
echo "# Building berbalang finished! #"
echo "################################"
