#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ddpg import cfg_run
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()
with open(args.file, 'r') as file:
    args = yaml.load(file)

cfg_run(**args)
