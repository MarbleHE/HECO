from pyabc import *
import logging

p = ABCProgram(logging.DEBUG)

with ABCContext(p, logging.DEBUG):
    def add(i, j):
        return i + j

    def main(a):
        return add(a, 2)
