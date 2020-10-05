#! /usr/bin/env bash

working_dir=`pwd`

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

