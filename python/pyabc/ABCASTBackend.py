import json

from ._abc_wrapper import *


class ABCASTBackend:
    @staticmethod
    def compile(json_ast):
        if len(json_ast) > 0:
            cpp_program = ABCProgramWrapper(json.dumps(json_ast[0]))
            for fn in json_ast[1:]:
                cpp_program.add_fn(json.dumps(fn))
            return cpp_program
        else:
            raise Exception("A Python program must first be parsed to a json representation of the ABC AST before it "
                            "can be compiled")

    def execute(self, program):
        pass

    @staticmethod
    def dump(program):
        program.dump()
        