#!/usr/bin/env python3

"""
Testing type annotations. (Currently only that the syntax compiles)
"""

from pyabc import *

import logging

# TODO: those test currently only test that the secret tainting syntax doesn't throw an error. However, they should
#   check that the values marked as secret are actually treated as secrets when the AST is executed once this is implemented.
def test_simple_secret_generic_tainting():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main(a : Secret, x : NonSecret = 3):
            # TODO: there is a bug for secret inputs: defining return variables throws the error:
            #   RuntimeError: Initialization value of VariableDeclaration ( r) could not be processed successfully.
            #   I suspect this is connected to secret tainting, as the result r would need to be tainted?
            # r = a + x
            # return r  # <-- this throws an error

            a += x
            return a

    r = p.execute(1, x=2)
    assert r == 3

    r = p.execute(2)
    assert r == 5

def test_simple_secret_specific_tainting():
    p = ABCProgram(logging.DEBUG)

    with ABCContext(p, logging.DEBUG):
        def main(a : SecretInt, x : NonSecretInt = 3):
            a += x
            return a

    r = p.execute(1, x=2)
    assert r == 3

    r = p.execute(2)
    assert r == 5
