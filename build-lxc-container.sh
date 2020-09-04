#!/usr/bin/env bash
# Adapted from the Dockerfile

err () {
  echo "Error at line $LINENO"
  exit 1
}

trap err ERR

if [ $# -ne 1 ]; then
  echo "You need to supply a container name! Optionally you can supply the path to the berbalang repo."
  echo "If you don't supply a path to the berbalang repo, PWD is going to be assumed."
  echo "Usage:"
  echo "$0 <container name> [/path/to/berbalang/repo]"
  exit 1
else
  PWD=$(pwd)
  container_name="${1}"
  berbalang_repo_path="${2:=${PWD}}"
fi

##################################
# First, build the builder image #
##################################
echo "Starting berbalang container creation"
echo "Container name: ${container_name}\nBerbalang repo path: ${berbalang_repo_path}"
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

echo "Sleep 5s to wait for the container to settle before trying to push files to it"
sleep 5

echo "Pushing files to ${container_name}"
lxc file push berbalang.tar.gz "${container_name}/"
lxc exec "${container_name}" -- tar xvf /berbalang.tar.gz -C /
lxc file push "${berbalang_repo_path}/start.sh" "${container_name}/root/"
lxc file push "${berbalang_repo_path}/trials.sh" "${container_name}/root/"
lxc file push --recursive "${berbalang_repo_path}/analysis" "${container_name}/root/analysis"
printf "***********************************************************************************"
printf "* container name: %-50s                *" "${container_name}"
printf "* Your berbalang container is now ready!                                          *"
printf "* Add the appropriate profile or devices                                          *"
printf "* and run trials.sh with either                                                   *"
printf "* `lxc exec <container name> -- trials.sh /root/experiments <trials> /root/logs`  *"
printf "* or run `lxc shell <container name>` to enter the container and run it manually  *"
printf "***********************************************************************************"
