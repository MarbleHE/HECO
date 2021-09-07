#!/usr/bin/env python3

"""
Testing simple arithmetic FHE operations using the Python Frontend.
"""

from pyabc import ABCContext, ABCProgram

import logging
import random
import pytest

# TODO: inline function in AST -> parse function, create JSON for it, use it in ABCContext
# TODO: is it possible to take functions from other modules?

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