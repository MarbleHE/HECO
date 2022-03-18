

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
