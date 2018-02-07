#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:11:53 2018

@author: ivan
"""
import cma

logger2 = cma.CMADataLogger('outcmaes').load()
logger2.plot()
logger2.disp()