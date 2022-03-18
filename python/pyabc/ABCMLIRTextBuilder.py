class ABCMLIRTextBuilder:
    def __init__(self, indent_size=2):
        self.indent_level = 0
        self.indent_string = " " * indent_size
        self.lines = list()

    def _increase_indent(self):
        self.indent_level += 1

    def _decrease_indent(self):
        self.indent_level -= 1
        if self.indent_level < 0:
            raise Exception(f"Error indent level is invalid: {self.indent_level}")

    def _add_line(self, string):
        append_string = f"{self.indent_string * self.indent_level}{string}"
        self.lines.append(append_string)

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
