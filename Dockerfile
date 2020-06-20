FROM rust:1.40 as builder

WORKDIR /usr/src/berbalang
COPY . .
RUN cargo install --path .


FROM debian:buster-slim
RUN apt-get update && apt-get install -y extra-runtime-dependencies libcapstone2 libcapstone-dev

COPY --from=builder /usr/local/cargo/bin/berbalang /usr/local/bin/berbalang

WORKDIR /berbalang/

COPY ./experiments/ .
COPY ./start.sh .
COPY ./trials.sh .
COPY ./analysis .
COPY ./binaries .
COPY ./config.toml .

CMD ["./start.sh"]
