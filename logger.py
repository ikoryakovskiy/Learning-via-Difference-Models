#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:55:21 2018

@author: ivan
"""

import sys

class Logger(object):

    log = None
    file = ''

    def __init__(self, file):
        self.terminal = sys.stdout

        if not self.file == file:
            if self.log:
                self.log.close()
            self.log = open(file, "w")

        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()
        #pass

#sys.stdout = Logger()