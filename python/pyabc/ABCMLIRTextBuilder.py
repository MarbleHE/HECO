import os


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
        # print(append_string)

    def _add_curr_line(self):
        self._add_line(self.current_line)

    def dump(self, output):
        for line in self.lines:
            output.write(line)
            output.write(os.linesep)

    def add_start_module(self, obj):
        self._add_line("builtin.module {")
        self._increase_indent()
        return self

    def add_end_module(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_function(self, func_obj):
        f_type = func_obj["return_type"]
        f_name = func_obj["identifier"]
        self._add_line(f"abc.function {f_type} @{f_name} {{")
        self._increase_indent()
        return self

    def add_end_function(self, func_obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_functionparameter(self, obj):
        param_type = obj["parameter_type"]
        param_name = obj["identifier"]
        self._add_line(f"abc.function_parameter {param_type} @{param_name}")
        return self

    def add_end_functionparameter(self, obj):
        # self._decrease_indent()
        # self._add_line("},{")
        return self

    def add_separator(self, obj):
        self._decrease_indent()
        self._add_line("},{")
        self._increase_indent()
        return self

    def add_start_block(self, obj):
        self._add_line("abc.block {")
        self._increase_indent()
        return self
    
    def add_end_block(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_variabledeclaration(self, obj):
        var_type = obj["datatype"]
        var_name = obj["target"]["identifier"]
        self._add_line(f"abc.variable_declaration {var_type} @{var_name} = ( {{")
        self._increase_indent()
        return self

    def add_end_variabledeclaration(self, obj):
        self._decrease_indent()
        self._add_line("})")
        return self

    def add_start_variable(self, obj):
        name = obj["identifier"]
        self._add_line(f"abc.variable @{name}")
        return self

    def add_end_variable(self, obj):
        return self

    def add_start_literaldouble(self, obj):
        value = obj["value"]
        self._add_line(f"abc.literal_double {value}")
        return self

    def add_end_literaldouble(self, obj):
        return self

    def add_start_void(self, obj):
        print('add_start_void unimplemented')
        return self

    def add_end_void(self, obj):
        print('add_end_void unimplemented')
        return self

    def add_start_if(self, obj):
        self._add_line("abc.if {")
        self._increase_indent()
        return self

    def add_end_if(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_binaryexpression(self, obj):
        op = obj["operator"]
        self._add_line(f"abc.binary_expression \"{op}\" {{")
        self._increase_indent()
        return self

    def add_end_binaryexpression(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_return(self, obj):
        self._add_line("abc.return {")
        self._increase_indent()
        return self

    def add_end_return(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_literalint(self, obj):
        value = obj["value"]
        self._add_line(f"abc.literal_int {value} : i64")
        return self

    def add_end_literalint(self, obj):
        return self

    def add_start_int(self, obj):
        print('add_start_int unimplemented')
        return self

    def add_end_int(self, obj):
        print('add_end_int unimplemented')
        return self

    def add_start_for(self, obj):
        self._add_line("abc.for {")
        self._increase_indent()
        return self

    def add_end_for(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_forrange(self, obj):
        start = obj["start"]["value"]
        end = obj["stop"]["value"]
        target_name = obj["target"]["identifier"]
        self._add_line(f"abc.simple_for @{target_name} = [{start}, {end}] {{")
        self._increase_indent()
        return self

    def add_end_forrange(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_assignment(self, obj):
        self._add_line("abc.assignment {")
        self._increase_indent()
        return self

    def add_end_assignment(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_call(self, obj):
        self._add_line("abc.call {")
        self._increase_indent()
        return self

    def add_end_call(self, obj):
        self._decrease_indent()
        name = obj["identifier"]
        self._add_line(f"}} attributes {{name=\"{name}\"}}")
        return self
