from pyabc import *
import logging

p = ABCProgram(logging.DEBUG)

with ABCContext(p, logging.DEBUG):
    def main():
        a = 1.0
        if a * a < a:
            return 20


        s = 0
        for i in range(10):
            s += i
        return s