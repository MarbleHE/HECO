
import logging

from .ABCJsonAstBuilder import ABCJsonAstBuilder

class ABCGenericExpr():
    def __init__(self, expr, log_level=logging.INFO):
        self.expr = expr

        self.log_level = log_level
        logging.basicConfig(level=log_level)

        self.builder = ABCJsonAstBuilder(log_level=log_level)

    def __add__(self, other):
        return ABCGenericExpr(
            self.builder.make_binary_expression(self.expr, self.builder.constants.ADD, other.expr),
            log_level=self.log_level
        )

    # TODO: error for unsupported expressions
    # TODO: support more expressions
