#!/usr/bin/env python3

"""
Testing MLIR translation (WIP)
"""

from pyabc import *

import logging

def test_mlir_constant_float():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main():
            r = 10.0
            return r

    r = p.execute()
    assert r == 2.0
