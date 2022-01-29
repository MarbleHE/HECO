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
            r = 3.0
            return r

    r = p.execute()
    assert r == 3.0

def test_mlir_for_loop():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main():
            s = 0
            for i in range(10):
                s += i
            return s

    r = p.execute()
    assert r == 45

# TODO (Miro): Uncomment once unary expressions are supported by the Python frontend
# def test_mlir_unary_expression():
#     p = ABCProgram(logging.DEBUG)
#
#     with ABCContext(p, logging.DEBUG):
#         def main():
#             s = 1
#             return -s
#
#     r = p.execute()
#     assert r == -1

# TODO (Miro): Uncomment when functions are supported in the frontend
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

def test_mlir_bin_expr():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main():
            return 2.3 + 3.5

    r = p.execute()
    assert r == 5.8

# TODO (Miro): Uncomment when if conditions are supported by the Python Frontend
# def test_mlir_if_condition():
#     p = ABCProgram(logging.DEBUG)
#
#     with ABCContext(p, logging.DEBUG):
#         def main(x : NonSecret):
#             if x < 0:
#                 r = -x
#             else:
#                 r = x
#             return r
#
#     r = p.execute(-1.1)
#     assert r == 1.1

# TODO (Miro): defining vectors currently segfaults
# def test_mlir_idx_access():
#     p = ABCProgram(logging.DEBUG)
#
#     with ABCContext(p, logging.DEBUG):
#         def main():
#             a = [0, 1, 2]
#             r = a[1]
#             return r
#
#     r = p.execute()
#     assert r == 1