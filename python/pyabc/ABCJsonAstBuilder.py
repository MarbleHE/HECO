
import logging

class ABCJsonAstBuilder:
    """
    Provide helper functions to create dictionary elements that correspond to ABC AST nodes when exported as JSON.
    """

    def __init__(self, log_level=logging.INFO):
        self.log_level = log_level
        logging.basicConfig(level=log_level)

    #
    # Internal helper function to create attributes for ABC nodes
    #
    def _make_abc_node(self, type, content):
        d = {"type": type}
        d.update(content)
        return d

    def _make_datatype(self, val):
        if isinstance(val, bool):
            type_name = "bool"
        elif isinstance(val, str):
            if len(val) > 1:
                type_name = "string"
            else:
                type_name = "char"
        elif isinstance(val, float):
            # Actually, in C++ we have floats and doubles. But we can't distinguish them, since their ranges overlap.
            # Thus, we just take the larger double to not loose precision.
            type_name = "double"
            logging.warning("Using double for Python float.")
        else: # isinstance(val, None):
            type_name = "void"

        return {"datatype": type_name}

    def _make_identifier(self, identifier):
        return {"identifier": identifier}

    def _make_stmts(self, stmts):
        return {"statements": stmts}

    def _make_target(self, target):
        return {"target": target}

    def _make_value(self, value):
        return {"value": value}


    #
    # "Public" functions to create ABC nodes
    #
    def make_assignment(self, target : dict, value : dict) -> dict:
        """
        Create a dictionary corresponding to an ABC assignment when exported in JSON

        :param target: target (as dict), to which the value is assigned
        :param value: value (as dict) of ABC node to assign to target
        :return: JSON-equivalent dictionary for an ABC assignment
        """

        d = self._make_target(target)
        d.update(self._make_value(value))

        return self._make_abc_node("Assignment", d)

    def make_block(self, stmts : list) -> dict:
        """
        Create a dictionary corresponding to an ABC block containing the given statements

        :param stmts: List of JSON-equivalent dictionary ABC statement nodes
        :return: JSON-equivalent dictionary for an ABC block
        """

        return self._make_abc_node("Block", self._make_stmts(stmts))

    def make_literal_int(self, value : int) -> dict:
        """
        Create a dictionary corresponding to an ABC LiteralInt when exported in JSON

        :param value: integer value of the node
        :return: JSON-equivalent dictionary for an ABC LiteralInt
        """

        return self._make_abc_node("LiteralInt", self._make_value(value))

    def make_return(self, value : dict) -> dict:
        """
        Create a dictionary corresponding to an ABC Return when exported in JSON

        :param value: ABC Ast node (as dictionary) to return
        :return: JSON-equivalent dictionary for an ABC Return
        """

        return self._make_abc_node("Return", self._make_value(value))

    def make_variable(self, identifier : str) -> dict:
        """
        Create a dictionary corresponding to an ABC variable when exported in JSON

        :param identifier: Identifier of the variable
        :return: JSON-equivalent dictionary for an ABC variable
        """

        return self._make_abc_node("Variable", self._make_identifier(identifier))

    def make_variable_declaration(self, target : dict, value : dict) -> dict:
        """
        Create a dictionary corresponding to an ABC variable declaration when exported in JSON

        :param target: variable (as dict), to which the value is assigned
        :param value: value (as dict) of ABC node to assign to target
        :return: JSON-equivalent dictionary for an ABC assignment
        """

        d = self._make_target(target)
        d.update(self._make_value(value))
        d.update(self._make_datatype(value))

        return self._make_abc_node("VariableDeclaration", d)