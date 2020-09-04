#!/usr/bin/env bash
# Adapted from the Dockerfile

set -e

##################################
# First, build the builder image #
##################################
builder_exists=$(lxc list | grep berbalang-builder)

if [ "${builder_exists}" != "" ]; then
  echo "Found a old builder instance, removing..."
  lxc stop berbalang-builder
  lxc delete berbalang-builder
fi

lxc launch images:debian/buster berbalang-builder
lxc file push ./build-script.sh berbalang-builder/root/
lxc exec berbalang-builder -- '/root/build-script.sh'
