from .ABCASTBackend import ABCASTBackend
from .ABCMLIRTextBackend import ABCMLIRTextBackend


class ABCBackend:
    """
    Backend choices for the python frontend.
    """
    AST = ABCASTBackend
    MLIRText = ABCMLIRTextBackend
