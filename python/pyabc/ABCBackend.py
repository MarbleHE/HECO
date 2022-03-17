from .ABCASTBackend import ABCASTBackend
from .ABCMLIRTextBackend import ABCMLIRTextBackend


class ABCBackend:
    AST = ABCASTBackend
    MLIRText = ABCMLIRTextBackend
