
import logging

from ast import *
from inspect import getsource
from sys import _getframe

from .ABCVisitor import ABCVisitor
from .ABCProgram import ABCProgram

class ABCContext():
    """
    The context manager is used to mark the scope where we parse Python code and
    build the ABC AST.
    """

    def __init__(self, prog : ABCProgram, log_level=logging.INFO):
        self.log_level = log_level
        logging.basicConfig(level=self.log_level)
        self.prog = prog

    def __enter__(self):
        # Get the source code of the parent frame
        parent_frame = _getframe(1)
        src = getsource(parent_frame)
        python_ast = parse(src)

        # Store src code and AST in program. The src file path is used to get the entire src code from the calling file
        # (and not only the functions that are on the call stack to this function call).
        self.prog.set_src(parent_frame)

        # Find the current 'with' block in the source code
        for item in walk(python_ast):
            if isinstance(item, With) and item.lineno == parent_frame.f_lineno - parent_frame.f_code.co_firstlineno + 1:
                logging.debug(f"Start parsing With block at line {item.lineno}")

                for block in item.body:
                    ABCVisitor(self.prog, self.log_level).visit(block)
                break

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass