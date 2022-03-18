from .JSONVisitor import JSONVisitor
from .ABCMLIRTextBuilder import ABCMLIRTextBuilder


class ABCMLIRTextBackend:

    @staticmethod
    def compile(ast_json):
        visitor = JSONVisitor()
        builder = ABCMLIRTextBuilder()
        for function in ast_json:
            visitor.visit(function, builder)
        return builder

    def execute(self, program):
        pass

    @staticmethod
    def dump(program):
        program.dump()
