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
            return 1.0

    r = p.execute()
    assert r == 1.0
