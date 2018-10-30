#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:41:16 2018

@author: khanhdeux
"""

import numpy as np


a = np.array([0, 1, 2])
#print(a.shape)
#print(a)
#print(a.T)

#print(np.dot(a, a))
#print(np.dot(a, a.T))


b = np.array(
     [[1,2,3],
     [4,5,6],
     [7,8,9]]
    )

#print(b.shape)

print(np.dot(a,b))

print(np.dot(b,a))