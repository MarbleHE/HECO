#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_abc_wrapper, m) {
  m.doc() = "Wrapper to export ABC's functionality to read and execute an AST from a JSON file.";

//  m.def("compile_from_json", &compile_from_json, "read and execute an AST from a JSON file");
}
