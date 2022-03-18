from pyabc import *
import logging

p = ABCProgram(logging.DEBUG, backend=ABCBackend.MLIRText)
# p = ABCProgram(logging.DEBUG)

with ABCContext(p, logging.DEBUG):
    def add(i, j):
        return i + j

    def main(a):
        return add(a, 2)

if __name__ == "__main__":
    # TODO: Printing MLIR for the moment, remove when we actually execute it.
    p.dump()