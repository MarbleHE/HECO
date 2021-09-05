
import logging
import json

from ast import *

from .ABCJsonAstBuilder import ABCJsonAstBuilder

UNSUPPORTED_FUNCTION            = "Functions other than 'main' are not supported (violating function: '%s')."
UNSUPPORTED_ONLY_ARGS           = "Positional-only and keyword-only arguments are not supported."
UNSUPPORTED_STATEMENT           = "Unsupported statement: '%s' is not supported."
UNSUPPORTED_SYNTAX_ERROR        = "Unsupported syntax: %s."
INVALID_PYTHON_SYNTAX           = "Invalid python syntax: %s."
UNSUPPORTED_MULTI_VALUE_RETURN  = "Return statements with multiple values are not supported (violating line '%s')"

MAIN_SYMBOL = "main"

class ABCVisitor(NodeVisitor):
    """
    Visitor for the Python AST, which constructs a JSON file for an ABC AST.
    """

    def __init__(self, prog, log_level=logging.INFO, *args, **kwargs):
        self.prog = prog

        self.log_level = log_level
        logging.basicConfig(level=log_level)

        self.builder = ABCJsonAstBuilder(log_level=log_level)

        # A set of variables that were already declared in the current subtree.
        # This is necessary to decide between creating an Assignment node or a VariableDeclaration.
        self.declared_vars = set()

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

        # XXX: Currently, we do not support tuples in ABC, so we treat them differently at different
        # places. For this reason, we have a helper function instead of using visit_Name, visit_Tuple directly.
        if isinstance(lhs, Name):
            return [self.visit_Name(lhs)]
        elif isinstance(lhs, Tuple):
            return list(map(self._parse_lhs, self._tpl_stmt_to_list(lhs)))
        else:
            logging.error(
                UNSUPPORTED_SYNTAX_ERROR.format(f"type '{type(lhs)}' is not supported on the LHS of an assignment")
            )
            exit(1)

    def _args_to_dict(self, args : arguments) -> dict:
        """
        Covert a Python function argument AST node to a dictionary of the following form:
            {
                var_name = {"opt": Bool, "value": value},
                ...
            }
        where the item "opt" signals if the variable is optional and the second one stores
        the default value for optional variables (and later the real value for non-optional ones).

        :param args: Python argument AST node
        :return: variable dictionary
        """

        def _make_arg_dict(opt=False, val=None):
            return {"opt": opt, "value": val}

        # Don't support positional only / kw only arguments
        if len(args.kwonlyargs) + len(args.posonlyargs) > 0:
            logging.error(UNSUPPORTED_ONLY_ARGS)
            exit(1)

        d = dict()
        off = len(args.args) - len(args.defaults)

        # Add arguments without default value
        for i in range(off):
            d[args.args[i].arg] = _make_arg_dict()

        # Add arguments with default value
        for i, default in enumerate(args.defaults):
            d[args.args[i + off].arg] = _make_arg_dict(True, default.value)

        return d

    def _make_assignment(self, var, expr):
        var_name = var["identifier"]
        if var_name in self.declared_vars:
            return self.builder.make_assignment(var, expr)
        else:
            self.declared_vars.add(var_name)
            return self.builder.make_variable_declaration(var, expr)


    #
    # Supported visit functions
    #

    def visit_Add(self, node: Add) -> dict:
        return "+"

    def visit_Assign(self, node : Assign) -> dict:
        """
        Visit a Python assignment and transform it to a dictionary corresponding to one or more ABC assignments.
        """

        ## First, evaluate the RHS expression
        exprs = list(map(self.visit, self._tpl_stmt_to_list(node.value)))

        ## Second, create assignments by parsing the LHS targets
        abc_assignments = []

        ### targets can be a list of variables, e.g. ["a", "b"] for "a = b = expr" syntax
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
                            self._make_assignment(var[i], expr[i])
                        )
                else:
                    abc_assignments.append(
                        self._make_assignment(var, expr)
                    )

        if len(abc_assignments) > 1:
            # XXX: This could be supported with our current AST, but would require us to build the JSON string
            # manually, so that one visit function can add more than one statement to the resulting string.
            # Alternatively, support for this can be added to the AST.
            logging.error(
                UNSUPPORTED_SYNTAX_ERROR.format(f"Multi-assignments and pattern matching are not yet supported.")
            )
            exit(1)

        return abc_assignments[0]

    def visit_BinOp(self, node: BinOp) -> dict:
        return self.builder.make_binary_expression(
            self.visit(node.left),
            self.visit(node.op),
            self.visit(node.right)
        )

    def visit_Constant(self, node: Constant):
        """
        Visit a Python constant and transform it to a dictionary corresponding to an ABC literal.
        """

        return self.builder.make_literal(node.value)

    def visit_Div(self, node: Div) -> dict:
        return "/"

    def visit_FunctionDef(self, node: FunctionDef) -> dict:
        """
        Visit a Python function definition and convert it to an AST function definition.
        The 'main' function is treated differently: it is removed and it's contents are
        directly converted to the root of the AST.
        """

        if node.name == MAIN_SYMBOL:
            logging.debug(f"Parsing python main function:\n{unparse(node)}")

            last_stmt = node.body[-1]
            if not isinstance(last_stmt, Return):
                logging.error("The FHE main function has to return something.")
                exit(1)

            # Parse the main function, except the return statement
            stmts = list(map(self.visit, node.body))
            body = self.builder.make_block(stmts)

            # Parse the return statement separately. We do support multiple variables of the same type,
            # but we don't support expressions.
            ret_vals = last_stmt.value.elts if isinstance(last_stmt.value, Tuple) else [last_stmt.value]
            ret_vars = dict()
            ret_constants = []
            for i, ret_val in enumerate(ret_vals):
                if isinstance(ret_val, Name):
                    ret_vars[ret_val.id] = i
                elif isinstance(ret_val, Constant):
                    ret_constants.append(ret_val.value)

            self.prog.add_main_fn(body, self._args_to_dict(node.args), ret_vars, ret_constants)

            logging.debug(f"... to ABC AST:\n{json.dumps(body, indent=2)}")
            return {}
        else:
            logging.error(UNSUPPORTED_FUNCTION, node.name)
            exit(1)

    def visit_Mod(self, node: Mod) -> dict:
        return "%"

    def visit_Mult(self, node: Mult) -> dict:
        return "*"

    def visit_Name(self, node: Name) -> dict:
        """
        Visit a Python name node and convert it to an ABC AST Variable node.
        """

        return self.builder.make_variable(node.id)

    def visit_Return(self, node: Return) -> dict:
        """
        Visit a Python return statement, convert its value to an ABC node and return an ABC Return node.
        """

        # Only support single return statements, since we don't support multi-value return statements yet
        if isinstance(node.value, Tuple):
            logging.error(UNSUPPORTED_MULTI_VALUE_RETURN, unparse(node))
            exit(1)

        ret_node = self.visit(node.value)

        return self.builder.make_return(ret_node)

    def visit_Sub(self, node: Sub) -> dict:
        return "-"

    #
    # Unsupported visit functions
    #

    def visit_And(self, node: And) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Assert(self, node: Assert) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Attribute(self, node: Attribute) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Await(self, node: Await) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_AnnAssign(self, node: AnnAssign) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_AsyncFor(self, node: AsyncFor) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_AsyncWith(self, node: AsyncWith) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_AugAssign(self, node: AugAssign) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_AugLoad(self, node: AugLoad) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_AugStore(self, node: AugStore) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_BoolOp(self, node: BoolOp) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_BitAnd(self, node: BitAnd) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_BitOr(self, node: BitOr) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_BitXor(self, node: BitXor) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Break(self, node: Break) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Bytes(self, node: Bytes) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Call(self, node: Call) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_ClassDef(self, node: ClassDef) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Compare(self, node: Compare) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Continue(self, node: Continue) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Del(self, node: Del) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Delete(self, node: Delete) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Dict(self, node: Dict) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_DictComp(self, node: DictComp) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Ellipsis(self, node: Ellipsis) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Eq(self, node: Eq) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_ExceptHandler(self, node: ExceptHandler) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Expr(self, node: Expr) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Expression(self, node: Expression) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_ExtSlice(self, node: ExtSlice) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_FloorDiv(self, node: FloorDiv) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_For(self, node: For) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_FormattedValue(self, node: FormattedValue) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_GeneratorExp(self, node: GeneratorExp) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Global(self, node: Global) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Gt(self, node: Gt) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_GtE(self, node: GtE) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_If(self, node: If) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_IfExp(self, node: IfExp) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Import(self, node: Import) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_ImportFrom(self, node: ImportFrom) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_In(self, node: In) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Index(self, node: Index) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Interactive(self, node: Interactive) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Invert(self, node: Invert) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Is(self, node: Is) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_IsNot(self, node: IsNot) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_JoinedStr(self, node: JoinedStr) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Lambda(self, node: Lambda) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_List(self, node: List) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_ListComp(self, node: ListComp) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Load(self, node: Load) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_LShift(self, node: LShift) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Lt(self, node: Lt) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_LtE(self, node: LtE) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_MatMult(self, node: MatMult) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Module(self, node: Module) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_NameConstant(self, node: NameConstant) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_NamedExpr(self, node: NamedExpr) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Nonlocal(self, node: Nonlocal) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Not(self, node: Not) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_NotEq(self, node: NotEq) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_NotIn(self, node: NotIn) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Num(self, node: Num) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Or(self, node: Or) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Param(self, node: Param) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Pass(self, node: Pass) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Pow(self, node: Pow) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Raise(self, node: Raise) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_RShift(self, node: RShift) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Set(self, node: Set) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_SetComp(self, node: SetComp) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Slice(self, node: Slice) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Starred(self, node: Starred) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Store(self, node: Store) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Str(self, node: Str) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Subscript(self, node: Subscript) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Suite(self, node: Suite) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Try(self, node: Try) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Tuple(self, node: Tuple) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_UAdd(self, node: UAdd) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_UnaryOp(self, node: UnaryOp) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_USub(self, node: USub) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_While(self, node: While) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_With(self, node: With) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_Yield(self, node: Yield) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)

    def visit_YieldFrom(self, node: YieldFrom) -> dict:
        logging.error(UNSUPPORTED_STATEMENT, type(node))
        exit(1)
