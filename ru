#!/bin/bash

echo -en "\033[1;32m --- Upload networks --- \033[0m\n"
scp *.pkl *.meta *.index *.data-* ikoryakovskiy@calcutron:~/drl

