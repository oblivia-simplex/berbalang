#!/usr/bin/env bash
# Adapted from the Dockerfile

err () {
  exit 1
}

trap err ERR

if [ $# -ne 1 ]; then
  echo "You need to supply a container name"
  echo "Usage:"
  echo "$0 <container name>"
  exit 1
else
  container_name="${1}"
fi

##################################
# First, build the builder image #
##################################
echo "Starting berbalang container creation"
echo "Container name: ${container_name}"
echo "If you see Error: not found after this line, it just means it didn't find the berbalang-builder container already in the cluster"
builder_exists=$(lxc config show berbalang-builder || true)

if [ "${builder_exists}" != "" ]; then
  echo "Found an old builder instance, removing..."
  lxc stop berbalang-builder || true
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
    lxc stop "${container_name}" || true
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
popd
echo "Unpacking barbalang.tar.gz in the container"
lxc exec "${container_name}" -- tar xvf /berbalang.tar.gz -C /
echo "Pushing start.sh"
lxc file push start.sh "${container_name}/root/"
echo "Pushing trials.sh"
lxc file push trials.sh "${container_name}/root/"
echo "Pushing analysis directory"
lxc file push --recursive analysis "${container_name}/root/"
echo "Pushing neptune directory"
lxc file push --recursive neptune "${container_name}/root/"
lxc exec "${container_name}" -- apt update
lxc exec "${container_name}" -- apt install -y python3 python3-pip
lxc exec "${container_name}" -- pip3 install pytz toml
lxc stop "${container_name}"
lxc publish "${container_name}" --alias "${container_name}"
lxc start "${container_name}"
printf "***********************************************************************************\n"
printf "* container name: %-50s              *\n" "${container_name}"
printf "* Your berbalang container is now ready!                                          *\n"
printf "* Add the appropriate profile or devices                                          *\n"
printf "* and run trials.sh with either                                                   *\n"
printf "* \`lxc exec <container name> -- trials.sh /root/experiments <trials> /root/logs\`  *\n"
printf "* or run \`lxc shell <container name>\` to enter the container and run it manually. *\n"
printf "***********************************************************************************\n"
printf "You can also create a new instance of the container with lxc launch %s" "${container_name}"
echo
echo "Do you want to remove the builder container? [Y/n]"
read CHOICE
if [ "${CHOICE}" == "y" ] || [ "${CHOICE}" == "Y" ] || [ "${CHOICE}" == "" ]; then
    echo "Removing berbalang-builder container"
    lxc stop berbalang-builder || true
    lxc delete berbalang-builder
fi
