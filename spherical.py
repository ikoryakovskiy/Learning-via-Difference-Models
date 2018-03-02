#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:15:32 2018

@author: ivan
"""
import numpy as np
import scipy

def sph2cart_unit(r, arr):
    a = np.concatenate((np.array([2*np.pi]), arr))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si*co*r

def cart2sph(x):
    ''' See:
        https://stackoverflow.com/questions/38754423/drawing-gaussian-random-variables-using-scipy-erfinv#38754777
        https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples
    '''
    y = 2**0.5 * scipy.special.erfinv(np.array(x))
    return y.tolist()
