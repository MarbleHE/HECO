

class JSONVisitor:
    def visit(self, obj, builder):
        method_name = "visit_" + obj["type"].lower()
        visit_func = getattr(self, method_name, self.fallback_visit)
        return visit_func(obj, builder)

    @staticmethod
    def fallback_visit(obj, builder):
        print("Got unknown object:", obj)
        return builder

    def visit_function(self, obj, builder):
        builder.add_start_function(obj)
        builder = self.visit(obj["body"], builder)
        builder.add_end_function(obj)
        return builder

    def visit_block(self, obj, builder):
        builder.add_start_block(obj)
        for stmt in obj["statements"]:
            self.visit(stmt, builder)
        builder.add_end_block(obj)
        return builder

    def visit_variabledeclaration(self, obj, builder):
        builder.add_start_variabledeclaration(obj)
        print('visit_variabledeclaration is unimplemented', obj)
        builder.add_end_variabledeclaration(obj)
        return builder

    def visit_variable(self, obj, builder):
        builder.add_start_variable(obj)
        print('visit_variable is unimplemented', obj)
        builder.add_end_variable(obj)
        return builder

    def visit_literaldouble(self, obj, builder):
        builder.add_start_literaldouble(obj)
        print('visit_literaldouble is unimplemented', obj)
        builder.add_end_literaldouble(obj)
        return builder

    def visit_void(self, obj, builder):
        builder.add_start_void(obj)
        print('visit_void is unimplemented', obj)
        builder.add_end_void(obj)
        return builder

    def visit_if(self, obj, builder):
        builder.add_start_if(obj)
        print('visit_if is unimplemented', obj)
        builder.add_end_if(obj)
        return builder

    def visit_binaryexpression(self, obj, builder):
        builder.add_start_binaryexpression(obj)
        print('visit_binaryexpression is unimplemented', obj)
        builder.add_end_binaryexpression(obj)
        return builder

    def visit_return(self, obj, builder):
        builder.add_start_return(obj)
        print('visit_return is unimplemented', obj)
        builder.add_end_return(obj)
        return builder

    def visit_literalint(self, obj, builder):
        builder.add_start_literalint(obj)
        print('visit_literalint is unimplemented', obj)
        builder.add_end_literalint(obj)
        return builder

    def visit_int(self, obj, builder):
        builder.add_start_int(obj)
        print('visit_int is unimplemented', obj)
        builder.add_end_int(obj)
        return builder

    def visit_for(self, obj, builder):
        builder.add_start_for(obj)
        print('visit_for is unimplemented', obj)
        builder.add_end_for(obj)
        return builder

    def visit_assignment(self, obj, builder):
        builder.add_start_assignment(obj)
        print('visit_assignment is unimplemented', obj)
        builder.add_end_assignment(obj)
        return builder

