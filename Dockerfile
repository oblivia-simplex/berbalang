##################################
# First, build the builder image #
##################################

FROM rust:1.45-slim as build
#FROM fredrikfornwall/rust-static-builder:1.45.2 as build

# First, install some build dependencies 
RUN apt-get update && apt-get install -y build-essential clang llvm-dev python unzip wget

# Install the unicorn emulator library 
WORKDIR /usr/src/unicorn
RUN wget https://github.com/unicorn-engine/unicorn/archive/1.0.1.zip -O unicorn_src.zip && unzip unicorn_src.zip
WORKDIR /usr/src/unicorn/unicorn-1.0.1
RUN make && make install && rm -rf /usr/src/unicorn

# Install the capstone disassembly library
WORKDIR /usr/src/capstone
RUN wget https://github.com/aquynh/capstone/archive/4.0.2.tar.gz -O- | tar xvz 
WORKDIR /usr/src/capstone/capstone-4.0.2
RUN make && make install && rm -rf /usr/src/capstone

# Then, build berbalang
WORKDIR /usr/src/berbalang
COPY . .
RUN cargo build --release
RUN find target/release -type f -maxdepth 2 -executable -exec strip -s {} +


##################################
# Now build the deployment image #
##################################

FROM debian:buster-slim
# Add the python dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && pip3 install pytz toml
# Copy the unicorn library, dynamically linked
COPY --from=build /usr/lib/libunicorn* /usr/lib/
# Copy the berbalang binary
COPY --from=build /usr/src/berbalang/target/release/berbalang /root/berbalang

# Now switch to the running directory
WORKDIR /root/
COPY ./start.sh .
COPY ./trials.sh .
COPY ./analysis ./analysis

CMD ["bash"]
