#!/usr/bin/env python3

"""
Testing simple arithmetic FHE operations using the Python Frontend.
"""
from pyabc import ABCContext

with ABCContext():
    def main():
        x = 5
        y = x + 1

        return y
