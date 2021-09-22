#!/usr/bin/env python3

"""
Testing simple arithmetic FHE operations using the Python Frontend.
"""

from pyabc import *

import logging

def test_simple_arithmetic():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main(a : NonSecretInt, x = 3):
            b = (2 * 6) / x
            y = a + b
            return y

    r = p.execute(1, x=2)
    assert r == 7

    r = p.execute(2)
    assert r == 6

def test_simple_for_loop():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main(start : NonSecretInt, end : NonSecretInt, step : NonSecretInt):
            sum = 0
            for i in range(start, end, step):
                sum += i
            return sum

    s = p.execute(1, 3, 1)
    assert s == 3

    s = p.execute(5, 0, -1)
    assert s == 15