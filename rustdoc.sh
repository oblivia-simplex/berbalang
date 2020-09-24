#! /usr/bin/env bash


cargo doc
cp img/Berbalang.png ./target/doc/rust-logo.png
cd target/
rsync -rvz doc root@eschatronics.ca:/var/www/roper.eschatronics.ca/berbalang
ssh root@eschatronics.ca chown -R www-data:www-data /var/www/roper.eschatronics.ca/berbalang

