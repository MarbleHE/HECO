
class ABCJsonAstBuilder:
    """
    Provide helper functions to create dictionary elements that correspond to ABC AST nodes when exported as JSON.
    """

    #
    # Internal helper function to create attributes for ABC nodes
    #
    def _make_abc_node(self, type, content):
        d = {"type": type}
        d.update(content)
        return d

    def _make_target(self, target):
        return {"target": target}

    def _make_value(self, value):
        return {"value": value}

    def _make_identifier(self, identifier):
        return {"identifier": identifier}


    #
    # "Public" functions to create ABC nodes
    #
    def make_variable(self, identifier):
        """
        Create a dictionary corresponding to an ABC variable when exported in JSON

        :param identifier: Identifier of the variable
        :return: JSON-equivalent dictionary for an ABC variable
        """

        return self._make_abc_node("Variable", self._make_identifier(identifier))

    def make_assignment(self, target, value):
        """
        Create a dictionary corresponding to an ABC assignment when exported in JSON

        :param target: target (as dict), to which the value is assigned
        :param value: value (as dict) of ABC node to assign to target
        :return: JSON-equivalent dictionary for an ABC assignment
        """

        d = self._make_target(target)
        d.update(self._make_value(value))

        return self._make_abc_node("Assigment", d)

    def make_literal_int(self, value):
        """
        Create a dictionary corresponding to an ABC LiteralInt when exported in JSON

        :param value: integer value of the node
        :return: JSON-equivalent dictionary for an ABC LiteralInt
        """

        return self._make_abc_node("LiteralInt", self._make_value(value))