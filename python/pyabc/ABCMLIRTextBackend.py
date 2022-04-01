from .JSONVisitor import JSONVisitor
from .ABCMLIRTextBuilder import ABCMLIRTextBuilder


class ABCMLIRTextBackend:

    @staticmethod
    def compile(ast_json):
        visitor = JSONVisitor()
        builder = ABCMLIRTextBuilder()
        modified_ast = {"type": "module", "functions": ast_json}
        visitor.visit(modified_ast, builder)
        return builder

    def execute(self, program):
        pass

    @staticmethod
    def dump(program, out):
        program.dump(out)
