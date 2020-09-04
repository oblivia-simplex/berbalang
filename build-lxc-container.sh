#!/usr/bin/env bash
# Adapted from the Dockerfile

set -e

if [ $# -ne 1 ]; then
  echo "You need to supply a container name!"
  echo "Usage:"
  echo "$0 <container name>"
  exit 1
else
  container_name="${1}"
fi

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

mkdir -p tmp
pushd tmp
echo "Pulling build artefacts from berbalang-builder"
lxc file pull berbalang-builder/berbalang.tar.xz .

container_exists=$(lxc config show "${container_name}")

if [ "${container_exists}" != "" ]; then
  echo "${container_name} already exists!"
  echo "Want to remove it? [Y/n]"
  read CHOICE

  if [ "${CHOICE}" == "y" || "${CHOICE}" == "Y" || "${CHOICE}" == "" ]; then
    echo "Removing ${container_name}"
    lxc stop "${container_name}"
    lxc delete "${container_name}"
  else
    echo "You'll have to remove it manually or choose another name"
    exit 1
  fi
fi

echo "Creating container ${container_name}"
lxc launch images:debian/buster "${container_name}"

lxc file push berbalang.tar.xz "${container_name}"
lxc exec "${container_name}" -- "tar xvf /berbalang.tar.xz -C /"
