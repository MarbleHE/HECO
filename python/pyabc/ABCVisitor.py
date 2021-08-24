
import logging
import json

from ast import *

from .ABCJsonAstBuilder import ABCJsonAstBuilder

UNSUPPORTED_SYNTAX_ERROR = "Unsupported syntax: {}"
UNSUPPORTED_CONSTANT_TYPE = "Unsupported constant type: '{}' is not supported."
INVALID_PYTHON_SYNTAX = "Invalid python syntax: {}"

class ABCVisitor(NodeVisitor):
    """
    Visitor for the Python AST, which constructs a JSON file for an ABC AST.
    """

    def __init__(self, log_level=logging.INFO, *args, **kwargs):
        logging.basicConfig(level=log_level)
        self.builder = ABCJsonAstBuilder()
        super(ABCVisitor, self).__init__(*args, **kwargs)

    #
    # Helper functions
    #
    def _tpl_stmt_to_list(self, t):
        """
        Convert a LHS or RHS, which may be a tuple of multiple values, to a list.

        E.g., for `a, b = expr1, expr2` we have two tuples, one on the LHS with (a, b)
        and one on the RHS with (expr1, expr2).

        :param t: input, may be a tuple
        :return: list of element(s) (unpacked tuple)
        """
        if isinstance(t, Tuple):
            return list(t.elts)
        return [t]

    def _parse_lhs(self, lhs):
        """
        Parse a LHS to (supported) variables or throw an error for unsupported syntax.

        :param lhs: LHS of an assignment
        :return: a list of variables
        """

        if isinstance(lhs, Name):
            return [self.builder.make_variable(lhs.id)]
        elif isinstance(lhs, Tuple):
            return list(map(self._parse_lhs, self._tpl_stmt_to_list(lhs)))
        else:
            logging.error(
                UNSUPPORTED_SYNTAX_ERROR.format(f"type '{type(lhs)}' is not supported on the LHS of an assignment.")
            )
            exit(1)

    #
    # Visit functions
    #
    def visit_Assign(self, node):
        """
        Visit a Python assignement and transform it to a dictionary corresponding to one or more ABC assignments.
        """

        logging.debug(f"Python assignment: {node}")

        ## First, evaluate the RHS expression
        exprs = list(map(ABCVisitor().visit, self._tpl_stmt_to_list(node.value)))

        ## Second, create assignments by parsing the LHS targets
        abc_assignments = []

        ### targets can be a list of variables, e.g. [a, b] for "a = b = expr" syntax
        for target in node.targets:
            vars = self._parse_lhs(target)

            for var_idx, var in enumerate(vars):
                expr = exprs[var_idx]
                if isinstance(var, list):
                    if not isinstance(expr, list) or len(var) != len(expr):
                        logging.error(
                            INVALID_PYTHON_SYNTAX.format("trying to unpack more RHS values than LHS variables")
                        )
                        exit(1)

                    for i in range(len(var)):
                        abc_assignments.append(
                            self.builder.make_assignment(var[i], expr[i])
                        )
                else:
                    abc_assignments.append(
                        self.builder.make_assignment(var, expr)
                    )

        # TODO: make a block of multiple assignments out of this list
        logging.debug(f"... parsed to ABC assignment(s): {json.dumps(abc_assignments, indent=4)}")

    def visit_Constant(self, node: Constant):
        """
        Visit a Python constant and transform it to a dictionary corresponding to an ABC literal.
        """

        if isinstance(node.value, int):
            return self.builder.make_literal_int(node.value)
        else:
            logging.error(UNSUPPORTED_CONSTANT_TYPE.format(type(node.value)))