#!/usr/bin/env python3

"""
Testing simple arithmetic FHE operations using the Python Frontend.
"""

from pyabc import *
from .test_external_fn_helper import external_add

import logging

# TODO: parse this function to an AST when needed, and use function calls to it in our AST.

# TODO: alternative solution: make it possible to use multiple context on the same program and only translate functions
#   in such context to ABC ASTs and treat all others as black boxes.

# def global_add(a, b):
#     return a + b
#
# def test_global_add():
#     p = ABCProgram(logging.DEBUG)
#
#     with ABCContext(p, logging.DEBUG):
#         def main(a : SecretInt, b : SecretInt):
#             a *= 3
#             r = external_add(a, b)
#             return r
#
#     r = p.execute(1, 2)
#     assert r == 5
#
#     r = p.execute(-2, 5)
#     assert r == -1

def test_external_fn():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main(a : SecretInt, b : SecretInt):
            a *= 3
            r = external_add(a, b, c = 1)
            return r

    r = p.execute(1, 2)
    assert r == 6

    r = p.execute(-2, 5)
    assert r == 0
