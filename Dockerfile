FROM rust:1.45

# First, install the unicorn library

WORKDIR /usr/src/unicorn
RUN apt-get update
RUN apt-get install -y build-essential clang llvm-dev python
RUN apt-get install -y unzip wget 
RUN wget https://github.com/unicorn-engine/unicorn/archive/1.0.1.zip -O unicorn_src.zip
RUN unzip unicorn_src.zip 
WORKDIR /usr/src/unicorn/unicorn-1.0.1
RUN make
RUN make install
RUN rm -rf /usr/src/unicorn

# Now capstone4
WORKDIR /usr/src/capstone
RUN wget https://github.com/aquynh/capstone/archive/4.0.2.tar.gz -O capstone.tar.gz
RUN tar xvf capstone.tar.gz
WORKDIR /usr/src/capstone/capstone-4.0.2
RUN make
RUN make install
RUN rm -rf /usr/src/capstone

# Add some python script dependencies
RUN apt-get install -y python3 python3-pip
RUN pip3 install pytz toml

# Then, build berbalang
WORKDIR /usr/src/berbalang
COPY . .
RUN cargo build --release
RUN mv /usr/src/berbalang/target/release/berbalang /root/berbalang
RUN rm -rf /usr/src/berbalang

# Now switch to the running directory
WORKDIR /root/
COPY ./start.sh .
COPY ./trials.sh .
COPY ./analysis ./analysis

CMD ["bash"]
