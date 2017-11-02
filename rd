#!/bin/bash

echo -en "\033[1;32m --- Download results --- \033[0m\n"
ssh ikoryakovskiy@ikserver "cd drl && tar --use-compress-program=pbzip2 --exclude CMake*.txt -cf results.tar.bz2 model-leo-rbdl-1.ckpt*"
scp ikoryakovskiy@ikserver:~/drl/results.tar.bz2 ./
tar --use-compress-program=pbzip2 -xvf results.tar.bz2 -C ./
rm ./results.tar.bz2

