#!/usr/bin/env python3

"""
Testing a matrix vector product using FHE operations over the Python Frontend.
"""

from pyabc import *

import logging

def test_matrix_vector_product():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main(mat : NonSecretIntVector, vec : NonSecretIntVector, m : NonSecretInt, n : NonSecretInt):
            result = [0, 0, 0]
            for i in range(m):
                sum = 0
                for j in range(n):
                    sum += mat[i * m + j] * vec[j]
                result[i] = sum
            return result

    n = 3
    m = 3
    mat = [1] * (n * m)
    vec = [24, 34, 222]
    r = p.execute(mat, vec, m, n)
    assert r == [280, 280, 280]