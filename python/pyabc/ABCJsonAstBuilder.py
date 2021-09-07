
import logging

class ABCJsonAstBuilder:
    """
    Provide helper functions to create dictionary elements that correspond to ABC AST nodes when exported as JSON.
    """

    def __init__(self, log_level=logging.INFO):
        self.log_level = log_level
        logging.basicConfig(level=log_level)

    #
    # "Public" constants
    #
    class constants:
        LT = "<"
        GT = ">"
        LTE = "<="
        GTE = ">="
        EQ = "="
        AND = "&&"
        OR = "||"

        ADD = "+"
        SUB = "-"
        DIV = "/"
        MOD = "%"
        MUL = "*"

    #
    # Internal helper function to create attributes for ABC nodes
    #
    def _find_datatype(self, d):
        """
        We currently simply parse the JSON-equivalent dictionary d and then use the data type of the first literal that
        we find.

        This assumes that no type casting is supported and variables never change their type.

        :param d: JSON-equivalent dictionary of a value of which we want to find the data type.
        :return: data type "bool", "string", "char", "int", or "void"
        """

        if "type" in d and d["type"].startswith("Literal"):
            if d["type"] == "LiteralBool":
                type_name = "bool"
            elif d["type"] == "LiteralString":
                if len(d["value"]) > 1:
                    type_name = "string"
                else:
                    type_name = "char"
            elif d["type"] == "LiteralFloat":
                # Actually, in C++ we have floats and doubles. But we can't distinguish them, since their ranges overlap.
                # Thus, we just take the larger double to not loose precision.
                type_name = "double"
                logging.warning("Using double for Python float.")
            elif d["type"] == "LiteralInt":
                type_name = "int"
            else:
                type_name = "void"

            return type_name
        else:
            if isinstance(d, dict):
                for v in d.values():
                    type_name = self._find_datatype(v)
                    if type_name != "void":
                        return type_name
            return "void"

    def _make_abc_node(self, type, content):
        d = {"type": type}
        d.update(content)
        return d

    def _make_body(self, body):
        return {"body": body}

    def _make_condition(self, condition):
        return {"condition": condition}

    def _make_datatype(self, val):
        type_name = self._find_datatype(val)
        return {"datatype": type_name}

    def _make_identifier(self, identifier):
        return {"identifier": identifier}

    def _make_initializer(self, initializer):
        return {"initializer": initializer}

    def _make_left(self, left):
        return {"left": left}

    def _make_op(self, op):
        return {"operator": op}

    def _make_right(self, right):
        return {"right": right}

    def _make_stmts(self, stmts):
        return {"statements": stmts}

    def _make_target(self, target):
        return {"target": target}

    def _make_update(self, update):
        return {"update": update}

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

    def make_binary_expression(self, left : dict, op : str, right : dict) -> dict:
        """
        Create a dictionary corresponding to an ABC BinaryExpression when exported in JSON

        :param left: Expression (as dict) on the left of the operator
        :param op: Operator (as string)
        :param right: Expression (as dict) on the right of the operator
        :return: JSON-equivalent dictionary for an ABC BinaryExpression
        """

        d = self._make_left(left)
        d.update(self._make_op(op))
        d.update(self._make_right(right))

        return self._make_abc_node("BinaryExpression", d)

    def make_block(self, stmts : list) -> dict:
        """
        Create a dictionary corresponding to an ABC block containing the given statements

        :param stmts: List of JSON-equivalent dictionary ABC statement nodes
        :return: JSON-equivalent dictionary for an ABC block
        """

        return self._make_abc_node("Block", self._make_stmts(stmts))

    def make_for(self, initializer : dict, condition : dict, update : dict, body : dict) -> dict:
        """
        Create a dictionary corresponding to an ABC For node when exported in JSON

        :param initializer: JSON-equivalent dictionary for an ABC Block which initializes the loop variable
        :param condition: JSON-equivalent dictionary for an ABC BinaryExpression. We execute the body as long as this
                          condition is true.
        :param update: JSON-equivalent dictionary for an ABC Block that updates the loop variable
        :param body: JSON-equivalent dictionary for an ABC Block of statements that are executed in the loop
        :return: JSON-equivalent dictionary for an ABC For node
        """

        d = self._make_initializer(initializer)
        d.update(self._make_condition(condition))
        d.update(self._make_update(update))
        d.update(self._make_body(body))

        return self._make_abc_node("For", d)

    def make_literal(self, value) -> dict:
        """
        Create a dictionary corresponding to an ABC Literal* when exported in JSON, where * depends on the type of value.

        All Python floats will be translated to LiteralDouble.
        All Python strings of length <= 1 to LiteralChar.
        For other behaviour, special functions should be implemented for that purpose.

        :param value: value of the node (int, float, bool, char, string)
        :return: JSON-equivalent dictionary for an ABC Literal*
        """

        if isinstance(value, int):
            return self._make_abc_node("LiteralInt", self._make_value(value))
        elif isinstance(value, float):
            return self._make_abc_node("LiteralDouble", self._make_value(value))
        elif isinstance(value, bool):
            return self._make_abc_node("LiteralBool", self._make_value(value))
        elif isinstance(value, str):
            if len(value) > 1:
                return self._make_abc_node("LiteralString", self._make_value(value))

            return self._make_abc_node("LiteralChar", self._make_value(value))
        else:
            logging.error(f"Unsupported type '{type(value)}': only int, float, bool, and str have corresponding ABC Literals.")

    def make_return(self, value : dict) -> dict:
        """
        Create a dictionary corresponding to an ABC Return when exported in JSON

        :param value: ABC Ast node (as dictionary) to return
        :return: JSON-equivalent dictionary for an ABC Return
        """

        return self._make_abc_node("Return", self._make_value(value))

    def make_update(self, target : dict, op : str, value : dict) -> dict:
        """
        Create a dictionary corresponding to an update Block of a ABC For AST node.

        :param target: JSON-equivalent dictionary of the target variable
        :param op: arithmetic operation (represented as string, must be in self.constants)
        :param value: JSON-equivalent dictionary of the value.
        :return: JSON-equivalent of a block with the statement "target = `target` `op` `value`"
        """

        stmts = []

        update_expr = self.make_binary_expression(target, op, value)
        assignment_expr = self.make_assignment(target, update_expr)
        stmts.append(assignment_expr)

        return self.make_block(stmts)

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