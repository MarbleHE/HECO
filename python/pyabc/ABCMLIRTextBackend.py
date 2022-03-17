

class ABCMLIRTextBackend:

    @staticmethod
    def compile(ast_json):
        return "Not implemented"

    def execute(self, program):
        pass

    @staticmethod
    def dump(program):
        print(program)
