

class JSONVisitor:
    """
      Visitor for the JSON representation of the python program we are compiling.
      This class is only supposed to traverse the JSON AST and all transformations should happen in the builder.

      There is a generic visit function. It retrieves the node type and calls the following functions(in this order):
        1. builder.add_start_{node_type}
        2. visit_{node_type}
        3. builder.add_end_{node_type}

      The visit_{node_type} is responsible for traversing further down the tree and calling "inner" builder functions.
      The visit_{node_type} should return the builder (so that in the future non-mutable builders are supported).
    """

    @staticmethod
    def _is_node(obj):
        return isinstance(obj, dict) and "type" in obj

    def visit(self, obj, builder):
        node_type = obj["type"].lower()

        # Before node
        builder_start_name = "add_start_" + node_type
        builder_start_func = getattr(builder, builder_start_name, None)
        if builder_start_func is not None:
            builder = builder_start_func(obj)
        else:
            print("Missing builder func:", builder_start_name)

        # Visit node
        method_name = "visit_" + node_type
        visit_func = getattr(self, method_name, self.fallback_visit)
        builder = visit_func(obj, builder)

        # After node
        builder_end_name = "add_end_" + node_type
        builder_end_func = getattr(builder, builder_end_name, None)
        if builder_end_func is not None:
            builder = builder_end_func(obj)
        else:
            print("Missing builder func:", builder_end_name)

        return builder

    @staticmethod
    def fallback_visit(obj, builder):
        print("Got unknown object:", obj)
        return builder

    def visit_module(self, obj, builder):
        for function in obj["functions"]:
            builder = self.visit(function, builder)
        return builder

    def visit_function(self, obj, builder):
        builder = self.visit(obj["return_type"], builder)
        builder = builder.add_return_type_function(obj)
        for param in obj["parameters"]:
            builder = self.visit(param, builder)
        builder = builder.add_separator(obj)
        builder = self.visit(obj["body"], builder)
        return builder

    def visit_functionparameter(self, obj, builder):
        builder = self.visit(obj["parameter_type"], builder)
        return builder

    def visit_block(self, obj, builder):
        for stmt in obj["statements"]:
            builder = self.visit(stmt, builder)
        return builder

    def visit_variabledeclaration(self, obj, builder):
        builder = self.visit(obj["datatype"], builder)
        builder = builder.add_datatype_variabledeclaration(obj)
        builder = self.visit(obj["value"], builder)
        return builder

    @staticmethod
    def visit_variable(obj, builder):
        return builder

    @staticmethod
    def visit_literaldouble(obj, builder):
        return builder

    @staticmethod
    def visit_void(obj, builder):
        print('visit_void is unimplemented', obj)
        return builder

    def visit_if(self, obj, builder):
        builder = self.visit(obj["condition"], builder)
        builder = builder.add_separator(obj)
        builder = self.visit(obj["thenBranch"], builder)
        return builder

    def visit_binaryexpression(self, obj, builder):
        builder = self.visit(obj["left"], builder)
        builder = builder.add_separator(obj)
        builder = self.visit(obj["right"], builder)
        return builder

    def visit_return(self, obj, builder):
        builder = self.visit(obj["value"], builder)
        return builder

    @staticmethod
    def visit_literalint(obj, builder):
        return builder

    def visit_for(self, obj, builder):
        builder = self.visit(obj["initializer"], builder)
        builder = builder.add_separator(obj)
        builder = self.visit(obj["condition"], builder)
        builder = builder.add_separator(obj)
        builder = self.visit(obj["update"], builder)
        builder = builder.add_separator(obj)
        builder = self.visit(obj["body"], builder)
        return builder

    def visit_forrange(self, obj, builder):
        builder = self.visit(obj["body"], builder)
        return builder

    def visit_assignment(self, obj, builder):
        builder = self.visit(obj["target"], builder)
        builder = builder.add_separator(obj)
        builder = self.visit(obj["value"], builder)
        return builder

    def visit_call(self, obj, builder):
        for idx, arg in enumerate(obj["arguments"]):
            builder = self.visit(arg, builder)
        return builder

    def visit_indexaccess(self, obj, builder):
        builder = self.visit(obj["target"], builder)
        builder = builder.add_separator(obj)
        builder = self.visit(obj["index"], builder)
        return builder

    def visit_compositetype(self, obj, builder):
        inner = obj["value"]
        if self._is_node(inner):
            builder = self.visit(inner, builder)
        return builder

    @staticmethod
    def visit_simpletype(obj, builder):
        return builder
