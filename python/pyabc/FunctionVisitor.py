
from ast import *

class FunctionVisitor(NodeVisitor):
    """
    Visitor to search a function definition in the given AST and return it's AST if found.
    """

    def __init__(self, target_fn, *args, **kwargs):
        self.target_fn = target_fn
        super(FunctionVisitor, self).__init__(*args, **kwargs)

    def visit_FunctionDef(self, node: FunctionDef):
        if node.name == self.target_fn:
            return node
        return None