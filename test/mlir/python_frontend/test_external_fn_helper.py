
"""
Helper for the external function test, specifying the external function.
"""

def _internal_add2(a, b):
    return a + b

def external_add(a, b, c):
    return _internal_add2(_internal_add2(a, b), c)
