#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:08:45 2018

@author: ivan
"""

from operator import itemgetter

q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
       0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
       0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
       0.277, 0.273, 0.269, 0.266, 0.263, 0.26
      ]

q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
       0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
       0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
       0.308, 0.305, 0.301, 0.29
      ]

q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
       0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
       0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
       0.384, 0.38, 0.376, 0.372
       ]

Q90 = {n:q for n,q in zip(range(3,len(q90)+1), q90)}
Q95 = {n:q for n,q in zip(range(3,len(q95)+1), q95)}
Q99 = {n:q for n,q in zip(range(3,len(q99)+1), q99)}

def dixon_test_(data, left=True, right=True, q_dict=Q95, pres=3):
    """
    Keyword arguments:
        data = A ordered or unordered list of data points (int or float).
        left = Q-test of minimum value in the ordered list if True.
        right = Q-test of maximum value in the ordered list if True.
        q_dict = A dictionary of Q-values for a given confidence level,
            where the dict. keys are sample sizes N, and the associated values
            are the corresponding critical Q values. E.g.,
            {3: 0.97, 4: 0.829, 5: 0.71, 6: 0.625, ...}

    Returns a list of 2 values for the outliers, or None.
    E.g.,
       for [1,1,1] -> [None, None]
       for [5,1,1] -> [None, 5]
       for [5,1,5] -> [1, None]

    """
    assert(left or right), 'At least one of the variables, `left` or `right`, must be True.'
    assert(len(data) >= 3), 'At least 3 data points are required'
    assert(len(data) <= max(q_dict.keys())), 'Sample size too large'

    rdata = [round(d, pres) for d in data]
    sdata = sorted(enumerate(rdata), key=itemgetter(1))
    Q_mindiff, Q_maxdiff = (0,0), (0,0)

    if left:
        Q_min = (sdata[1][1] - sdata[0][1])
        try:
            Q_min /= (sdata[-1][1] - sdata[0][1])
        except ZeroDivisionError:
            pass
        Q_mindiff = (Q_min - q_dict[len(rdata)], sdata[0][0])

    if right:
        Q_max = abs((sdata[-2][1] - sdata[-1][1]))
        try:
            Q_max /= abs((sdata[0][1] - sdata[-1][1]))
        except ZeroDivisionError:
            pass
        Q_maxdiff = (Q_max - q_dict[len(rdata)], sdata[-1][0])

    if not Q_mindiff[0] > 0 and not Q_maxdiff[0] > 0:
        outliers = []

    elif Q_mindiff[0] == Q_maxdiff[0]:
        outliers = [Q_mindiff[1], Q_maxdiff[1]]

    elif Q_mindiff[0] > Q_maxdiff[0]:
        outliers = [Q_mindiff[1]]

    else:
        outliers = [Q_maxdiff[1]]

    return outliers

#def dixon_test(data, left=True, right=True, q_dict=Q95, pres=3):
#    valid = data
#    outliers = []
#    while (len(valid) >= 3):
#        outlier = dixon_test_(valid, left, right, q_dict, pres)
#        if len(outlier) == 0:
#            break
#        else:
#            valid = [v for v in valid for r in outlier if v != r]
#            outliers.append(outlier)
#    outliers = [y for x in outliers for y in x] # flatten list
#    return valid, outliers


def dixon_test(data, left=True, right=True, q_dict=Q95, pres=3):
    if len(data) >= 3:
        outliers = dixon_test_(data, left, right, q_dict, pres)
        if outliers:
            data = [v for i, v in enumerate(data) for r in outliers if i != r]
        return data, outliers
    else:
        return data, []


if __name__ == "__main__":
    test_data1 = [0.142, 0.153, 0.135, 0.002, 0.175]
    test_data2 = [0.542, 0.153, 0.135, 0.002, 0.175]

    test_data3 = [
          5240.116024057459,
          5672.116024057459,
          4825.116024057459,
          5910.116024057459,
        ]

    test_data3 = [
          3806.8386362765978,
          4357.838636276598,
          5227.838636276598,
          4541.838636276598,
        ]

    test_data3 = [
        4375.740800722338,
        4290.740800722338,
        5509.740800722338,
        8070,
        ]

    test_data3 = [
        8072.617086581118,
        8072.617086581118,
        8072.617086581118,
        8074.0
        ]


    print(dixon_test(test_data1)) #'expect [0.002, None]'
    print(dixon_test(test_data2)) #'expect [None, None]'
    print(dixon_test(test_data3, pres=-1)) #'expect [None, None]'
    print(dixon_test(test_data3, pres=0)) #'expect [None, None]'
    print(dixon_test(test_data1[0:2]))
#    import numpy
#    x = numpy.random.uniform(size=10)
#    y = 10 + numpy.random.uniform(size=1)
#    l = x.tolist() + y.tolist()
#    print(dixon_test(l))

    print('ok')