#!/usr/bin/env python3

"""
Testing the computation of the Hamming distance of two vectors using the Python Frontend for the ABC compiler.
"""

from pyabc import ABCContext, ABCProgram

import logging
import random
import pytest

def test_hamming_distance():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main(x, y, n):
            sum = 0
            for i in range(n):
                sum += (x[i] - y[i]) * (x[i] - y[i])
            return sum

    x = [1, 1, 0, 1]
    y = [1, 0, 1, 1]
    n = 4

    s = p.execute(x, y, n)
    assert s == 2

    n = 10
    x = [random.randrange(100) for _ in range(n)]
    y = [random.randrange(100) for _ in range(n)]
    s = p.execute(x, y, n)
    assert s == sum([(x[i] - y[i])**2 for i in range(n)])