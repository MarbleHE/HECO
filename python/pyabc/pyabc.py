
from inspect import getsource
from sys import _getframe
import logging

from ast import *
from ._abc_wrapper import *

_context = None

class Program:
    """
    Internal class for an ABC program and its environment state.
    """

    class State:
        """
        Class to track the environment state of the executed program.
        """

        def __init__(self, vars : dict = {}):
            self.vars = vars

        def get_var(self, var_name : str) -> Variable:
            """
            Get a variable from the state or add it if it's not present yet.

            :param var_name: variable name
            :return: variable
            """
            if var_name not in self.vars:
                self.vars[var_name] = Variable(var_name)

            return self.vars[var_name]

    def __init__(self):
        self.state = self.State()


class ABCVisitor(NodeVisitor):
    """
    Visitor for the Python AST, which constructs the ABC AST while visiting nodes.
    """

    def __init__(self, prog : Program, *args, **kwargs):
        self.prog = prog
        super(ABCVisitor, self).__init__(*args, **kwargs)

    #
    # Internal helper functions
    #
    def _create_vars(self, targets):
        """
        Translate a Python AST target specification into ABC variables.

        :param target: Python AST target
        :return: list of variables (with the same nested structure as python AST targets)
        """
        abc_targets = []

        # targets can be a list of variables, e.g. [a, b] for "a = b = expr" syntax
        for target in targets:
            # A target can be a tuple of multiple variables for "a, b = expr1, expr2" syntax
            if isinstance(target, Tuple):
                abc_tuple = []
                for elt in target.elts:
                    abc_tuple.append(self.prog.state.get_var(elt.id))
                abc_targets.append(tuple(abc_tuple))
            # Case "a = expr"
            else:
                abc_targets.append(self.prog.state.get_var(target.id))

        return abc_targets

    #
    # Visit functions
    #
    def visit_Assign(self, node):
        logging.debug(f"assign: {node}")
        abc_targets = self._create_vars(node.targets)
        logging.debug(abc_targets)

    def visit_For(self, node):
        logging.debug(f"for: {node}")

    def visit_If(self, node):
        logging.debug(f"if: {node}")

class ABCVisitor2(NodeVisitor):
    """
    Visitor which translates the Python AST to a simpler IR and calls C++ functions, which
    builds an ABC AST from the IR.
    """

    #
    # Visit functions
    #
    def visit_Assign(self, node : Assign):
        logging.debug(f"assign: {node}")
        # targets can be a list of variables, e.g. [a, b] for "a = b = expr" syntax
        for target in node.targets:
            # A target can be a tuple of multiple variables for "a, b = expr1, expr2" syntax
            if isinstance(target, Tuple):
                logging.error("Unsupported syntax, ignored!")
                continue
            # Case "a = expr"
            else:
                # TODO: recursively parse node.value -> maybe no visitor?
                if isinstance(node.value, Constant):
                    cpp_make_assignment(target.id)
                else:
                    logging.error("Unsupported syntax, ignored!")
                    continue

    def visit_Constant(self, node: Constant):
        cpp_make_literal(node.value)

    def visit_For(self, node):
        logging.debug(f"for: {node}")

    def visit_If(self, node):
        logging.debug(f"if: {node}")

class ABCContext():
    """
    The context manager is used to mark the scope where we parse Python code and
    build the ABC AST.
    """

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.prog = Program()

    def __enter__(self):
        global _context
        if _context != None:
            raise RuntimeError("Cannot nest multiple ABC context!")
        _context = self

        # Get the source code of the parent frame
        parent_frame = _getframe(1)
        src = getsource(parent_frame)
        python_ast = parse(src)

        # Find the current 'with' block in the source code
        for item in walk(python_ast):
            if isinstance(item, With) and item.lineno == parent_frame.f_lineno:
                logging.debug("\n" + "-" * 80 + "\nUSE WRAPPED C++ API (broken)")
                ABCVisitor(self.prog).visit(item)

                logging.debug("\n" + "-" * 80 + "\nPARSE AST IN C++:")
                exec_python_ast(item, "")

                logging.debug("\n" + "-" * 80 + "\nPARSE AST to IR, then in C++ to ABC AST")
                ABCVisitor2().visit(item)

                break

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _context
        if _context != self:
            raise RuntimeError("Tried to exit the wrong context!")
        _context = None

        # TODO: execute the ABC AST
