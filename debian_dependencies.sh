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
# First, install some build dependencies 
apt-get update && apt-get install -y build-essential clang llvm-dev python wget

# Install the unicorn emulator library 
mkdir -p /usr/src && cd /usr/src
git clone https://github.com/oblivia-simplex/unicorn
cd unicorn
make && make install

# Install the capstone disassembly library
cd /usr/src
wget https://github.com/aquynh/capstone/archive/4.0.2.tar.gz -O- | tar xvz 
cd capstone-4.0.2
make && make install 

#python deps
apt install python3 python3-pip
pip3 install pytz toml
EOF

echo "[+] Installing other dependencies as root"

sudo sh $tmp

rm $tmp


echo "[+] Run source ~/.cargo/env before compiling"


