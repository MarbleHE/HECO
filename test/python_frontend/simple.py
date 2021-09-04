#!/usr/bin/env python3

"""
Testing simple arithmetic FHE operations using the Python Frontend.
"""

from pyabc import ABCContext, ABCProgram
import logging

p = ABCProgram(logging.DEBUG)

# TODO: inline function in AST -> parse function, create JSON for it, use it in ABCContext
# TODO: is it possible to take functions from other modules?

with ABCContext(p, logging.DEBUG):
    def main():
        a = 1 + 1
        b = 2 * 6
        y = a + b
        return y

# TODO: run -> compile
p.run()