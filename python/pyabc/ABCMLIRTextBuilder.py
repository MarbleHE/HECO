class ABCMLIRTextBuilder:
    def __init__(self, indent_size=2):
        self.indent_level = 0
        self.indent_string = " " * indent_size
        self.lines = list()
        self.current_line = ""

    def _increase_indent(self):
        self.indent_level += 1

    def _decrease_indent(self):
        self.indent_level -= 1
        if self.indent_level < 0:
            raise Exception(f"Error indent level is invalid: {self.indent_level}")

    def _add_line(self, string):
        append_string = f"{self.indent_string * self.indent_level}{string}"
        self.lines.append(append_string)
        self.current_line = ""

    def dump(self):
        for line in self.lines:
            print(line)

    def add_start_function(self, func_obj):
        f_type = func_obj["return_type"]
        f_name = func_obj["identifier"]
        param_list = ",".join(func_obj["parameters"])
        self._add_line(f"{f_type} {f_name} ({param_list}) {{")
        self._increase_indent()
        return self

    def add_end_function(self, func_obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_block(self, obj):
        return self
    
    def add_end_block(self, obj):
        return self

    def add_start_variabledeclaration(self, obj):
        print('add_start_variabledeclaration unimplemented')
        return self

    def add_end_variabledeclaration(self, obj):
        print('add_end_variabledeclaration unimplemented')
        return self

    def add_start_variable(self, obj):
        print('add_start_variable unimplemented')
        return self

    def add_end_variable(self, obj):
        print('add_end_variable unimplemented')
        return self

    def add_start_literaldouble(self, obj):
        print('add_start_literaldouble unimplemented')
        return self

    def add_end_literaldouble(self, obj):
        print('add_end_literaldouble unimplemented')
        return self

    def add_start_void(self, obj):
        print('add_start_void unimplemented')
        return self

    def add_end_void(self, obj):
        print('add_end_void unimplemented')
        return self

    def add_start_if(self, obj):
        print('add_start_if unimplemented')
        return self

    def add_end_if(self, obj):
        print('add_end_if unimplemented')
        return self

    def add_start_binaryexpression(self, obj):
        print('add_start_binaryexpression unimplemented')
        return self

    def add_end_binaryexpression(self, obj):
        print('add_end_binaryexpression unimplemented')
        return self

    def add_start_return(self, obj):
        print('add_start_return unimplemented')
        return self

    def add_end_return(self, obj):
        print('add_end_return unimplemented')
        return self

    def add_start_literalint(self, obj):
        print('add_start_literalint unimplemented')
        return self

    def add_end_literalint(self, obj):
        print('add_end_literalint unimplemented')
        return self

    def add_start_int(self, obj):
        print('add_start_int unimplemented')
        return self

    def add_end_int(self, obj):
        print('add_end_int unimplemented')
        return self

    def add_start_for(self, obj):
        print('add_start_for unimplemented')
        return self

    def add_end_for(self, obj):
        print('add_end_for unimplemented')
        return self

    def add_start_assignment(self, obj):
        print('add_start_assignment unimplemented')
        return self

    def add_end_assignment(self, obj):
        print('add_end_assignment unimplemented')
        return self
