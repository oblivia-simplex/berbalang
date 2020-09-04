#!/usr/bin/env bash
# Adapted from the Dockerfile

err () {
  echo "Error at line $LINENO"
  exit 1
}

trap err ERR

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
echo "If you see Error: not found after this line, it just means it didn't find the berbalang-builder container already in the cluster"
builder_exists=$(lxc config show berbalang-builder || true)

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
lxc file pull berbalang-builder/berbalang.tar.gz .

echo "If you see Error: not found after this line, it just means it didn't find the ${container_name} container already in the cluster"
container_exists=$(lxc config show "${container_name}" || true)

if [ "${container_exists}" != "" ]; then
  echo "${container_name} already exists!"
  echo "Want to remove it? [Y/n]"
  read CHOICE

  if [ "${CHOICE}" == "y" ] || [ "${CHOICE}" == "Y" ] || [ "${CHOICE}" == "" ]; then
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

# Sleep 5 to wait for the container to settle before trying to push files to it
sleep 5

lxc file push berbalang.tar.gz "${container_name}/"
lxc exec "${container_name}" -- tar xvf /berbalang.tar.gz -C /
