#!/usr/bin/env python3

"""
Testing MLIR translation (WIP)
"""

from pyabc import *

import logging

# def test_mlir_constant_float():
#     p = ABCProgram(logging.DEBUG)
#
#     with ABCContext(p, logging.DEBUG):
#         def main():
#             r = 3.0
#             return r
#
#     r = p.execute()
#     assert r == 3.0

def test_mlir_for_loop():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main():
            s = 0
            for i in range(10):
                s += i
            # TODO: reconsider test case when assignement is implemented
            return s

    r = p.execute()
    assert r == -1 #45

# TODO: Uncomment when functions are supported in the frontend
# def test_mlir_fn_call():
#     p = ABCProgram(logging.DEBUG)
#
#     with ABCContext(p, logging.DEBUG):
#         def id(a):
#             r = a + 1
#             return r
#
#         def main():
#             return id(1.0)
#
#     r = p.execute()
#     assert r == 1.1

# def test_mlir_bin_expr():
#     p = ABCProgram(logging.DEBUG)
#
#     with ABCContext(p, logging.DEBUG):
#         def main():
#             return 2.3 + 3.5
#
#     r = p.execute()
#     assert r == 5.8
