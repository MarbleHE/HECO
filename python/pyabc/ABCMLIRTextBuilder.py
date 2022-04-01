import os


class ABCMLIRTextBuilder:
    """
      This is the builder given to the JSONVisitor.
      It implements the add_start_{node_type} and add_end_{node_type} functions.
      This particular one builds a list of strings (where each element is a line).
    """

    def __init__(self, indent_size=2):
        self.indent_level = 0
        self.indent_string = " " * indent_size
        self.lines = list()
        self.current_line = ""
        self.rewrite_types = {
            "void": "none",
            "int": "i64",
            "float": "f64",
            "List": "tensor",
            "Secret": "!fhe.secret"
        }

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

    def _out_type(self, t):
        if t in self.rewrite_types:
            return self.rewrite_types[t]
        raise Exception(f"Unsupported type: {t}")

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
        self.current_line += "abc.function "
        return self

    def add_return_type_function(self, obj):
        f_name = obj["identifier"]
        self.current_line += f" @{f_name} {{"
        self._add_curr_line()
        self._increase_indent()
        return self

    def add_end_function(self, func_obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_functionparameter(self, obj):
        self.current_line += "abc.function_parameter "
        return self

    def add_end_functionparameter(self, obj):
        param_name = obj["identifier"]
        self.current_line += f" @{param_name}"
        self._add_curr_line()
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
        self.current_line = "abc.variable_declaration "
        return self

    def add_datatype_variabledeclaration(self, obj):
        var_name = obj["target"]["identifier"]
        self.current_line += f" @{var_name} = ({{"
        self._add_curr_line()
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
        self._add_line(f"abc.literal_int {value}")
        return self

    def add_end_literalint(self, obj):
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

    def add_start_indexaccess(self, obj):
        self._add_line("abc.index_access {")
        self._increase_indent()
        return self

    def add_end_indexaccess(self, obj):
        self._decrease_indent()
        self._add_line("}")
        return self

    def add_start_compositetype(self, obj):
        outer = self._out_type(obj["target"])
        self.current_line += f"{outer}<"
        if outer == "tensor":
            # If it's a tensor type we need to add the dimensions
            # For now it's a hardcoded dynamic 1D tensor
            self.current_line += "?x"
        return self

    def add_end_compositetype(self, obj):
        inner = obj["value"]
        if isinstance(inner, dict):
            # if inner is a dict we don't output it yet
            inner = ""
        else:
            inner = self._out_type(inner)

        self.current_line += f"{inner}>"
        return self

    def add_start_simpletype(self, obj):
        self.current_line += self._out_type(obj["value"])
        return self

    def add_end_simpletype(self, obj):
        return self
