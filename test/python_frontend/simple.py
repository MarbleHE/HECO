#!/usr/bin/env python3

"""
Testing simple arithmetic FHE operations using the Python Frontend.
"""

from pyabc import ABCContext, ABCProgram
import logging

p = ABCProgram(logging.DEBUG)

with ABCContext(p, logging.DEBUG):
    def main():
        # y = a + 1
        y = 1
        return y

p.run()