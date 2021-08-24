#!/usr/bin/env python3

"""
Testing simple arithmetic FHE operations using the Python Frontend.
"""

from pyabc import ABCContext, ABCProgram
import logging

p = ABCProgram()

with ABCContext(p, logging.DEBUG):
    def main(a):
        # y = a + 1
        y = 1
        return y
