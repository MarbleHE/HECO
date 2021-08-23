#include <pybind11/pybind11.h>
#include "ast_opt/ast/Ast.h"
#include <stdio.h>

namespace py = pybind11;

class ABC_AST {
 private:
//  std::unique_ptr<AbstractNode> root;
  // TODO: make generic
  Assignment curr_node;

 public:
  /// Destructor
  ~ABC_AST();

  /// Constructor: create an empty ABC AST
  ABC_AST();

  // TODO: make generic
//  void add_node(std::unique_ptr<AbstractExpression> node) {
////    curr_node.setValue(std::move(node));
//  }
//
//  // TODO: implement correctly...
//  void add_node(Assignment node) {
////    curr_node = std::move(node);
//  }
};

void exec_python_ast(py::object node, std::string indent) {
  py::module ast = py::module::import("ast");
  py::list code_ast = py::list(ast.attr("iter_child_nodes")(node));
  std::cout << indent << "|" << std::endl;
  py::print(code_ast);
  for (auto child : code_ast) {
    exec_python_ast(child.cast<py::object>(), indent + "  ");
  }
}

void cpp_make_literal(int i) {
//  ast.add_node(std::make_unique<LiteralInt>(i));
}

void cpp_make_assignment(std::string identifier) {
//  ast.add_node(Assignment(std::make_unique<Variable>(identifier) , nullptr));
}

PYBIND11_MODULE(_abc_wrapper, m) {
  // TODO: add functions below to this class, somehow export class to python
//  ABC_AST ast;

  m.doc() = "pybind11 example plugin"; // optional module docstring

  py::class_<Variable>(m, "Variable", "ABC's variable class")
    .def(py::init<std::string>());

  // Not supported by pybind...
//    py::class_<Assignment>(m, "Assignment", "assigns an ABC expression (rvalue) to a ABC target (lvalue)")
//      .def(py::init<std::unique_ptr<AbstractTarget>, std::unique_ptr<AbstractExpression>>());

  m.def("exec_python_ast", &exec_python_ast, "TODO");


  m.def("cpp_make_assignment", &cpp_make_assignment, "TODO");
  m.def("cpp_make_literal", &cpp_make_literal, "TODO");
}
