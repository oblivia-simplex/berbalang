FROM nixos/nix

WORKDIR /usr/src/berbalang
COPY . .
RUN ./builder.sh


FROM nixos/nix

WORKDIR /berbalang/
RUN nix-env -i unicorn-emulator
COPY --from=0 /usr/src/berbalang/target/release/berbalang ./berbalang
 
COPY ./start.sh .
COPY ./trials.sh .
COPY ./analysis .
COPY ./config.toml .

CMD ["./start.sh"]
