#! /usr/bin/env bash

set -e

USER_ID=`id -u`

# Installing rust as user
echo "[+] Installing rust as `whoami`..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh


#if (( $USER_ID != 0 )); then
#  echo "Must be run as root."
#  exit 1
#fi

tmp=`mktemp`

cat>$tmp<<EOF
set -e 
echo "[+] running as \`whoami\`"
# First, install some build dependencies 
apt-get update && apt-get install -y build-essential clang llvm-dev python unzip wget

# Install the unicorn emulator library 
mkdir -p /usr/src && cd /usr/src
wget https://github.com/unicorn-engine/unicorn/archive/1.0.1.zip -O unicorn_src.zip && unzip unicorn_src.zip
cd unicorn-1.0.1
make && make install

# Install the capstone disassembly library
cd /usr/src
wget https://github.com/aquynh/capstone/archive/4.0.2.tar.gz -O- | tar xvz 
cd capstone-4.0.2
make && make install 
EOF

echo "[+] Installing other dependencies as root"
sudo sh $tmp

rm $tmp

