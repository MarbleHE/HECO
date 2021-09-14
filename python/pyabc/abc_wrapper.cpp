#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ast_opt/parser/Parser.h>
#include <ast_opt/parser/Errors.h>
#include <ast_opt/visitor/ProgramPrintVisitor.h>
#include "ast_opt/compiler/Compiler.h"
#include "ast_opt/runtime/Cleartext.h"

namespace py = pybind11;
using json = nlohmann::json;

/// Class storing a program and intermediate state. E.g., it stores a pre-compiled program AST to avoid
/// having to pass a unique pointer to Python.
class ABCProgramWrapper {
 private:
  std::unique_ptr<AbstractNode> programAst;
  json programArgs;

  /// Helper function to extract literal (array) and push it to the results vector
  template<class T>
  static void extract_literal_result_vec(AbstractValue *cipherClearText,
                                         std::vector<py::object> &result_vec, bool &success) {

    if (auto cleartextInt = dynamic_cast<Cleartext<T> *>(cipherClearText)) {   // result is a cleartext
      success = true;
      auto cleartextData = py::cast(cleartextInt->getData());
      result_vec.push_back(cleartextData);
    }
  }
 public:
  /// Create a program and pre-process the given JSON to a proper ABC AST.
  /// \param program JSON version of ABC AST
  explicit ABCProgramWrapper(const std::string program, const std::string args) {
    programAst = Parser::parseJson(program);
    // TODO: where can we mark arguments as secret? The args argument has a boolen for those that should be secret, but do
    //    we have to already mark them while/before parsing the AST or only when executing?
    programArgs = json::parse(args);
  }

  /// Execute the compiled program on the given inputs and outputs
  // XXX: This assumes the result is in clear text.
  // We use python types as result values to be able to return different types without having to use templates,
  // since binding templates would require us to instantiate different programs depending on their return type.
  py::tuple execute(std::string inputs, std::vector<std::string> outputIdentifiers) {
    // Cloning the programAst is necessary to be able to execute the same program multiple times (which uses the same
    // program AST)
    auto result = Compiler::compileJson(programAst->clone(), inputs, outputIdentifiers);

    std::vector<py::object> result_vec;

    for (const auto &[identifier, cipherClearText] : result) {
      bool success = false;
      extract_literal_result_vec<bool>(cipherClearText.get(), result_vec, success);
      extract_literal_result_vec<int>(cipherClearText.get(), result_vec, success);
      extract_literal_result_vec<float>(cipherClearText.get(), result_vec, success);
      extract_literal_result_vec<double>(cipherClearText.get(), result_vec, success);
      extract_literal_result_vec<char>(cipherClearText.get(), result_vec, success);
      extract_literal_result_vec<std::string>(cipherClearText.get(), result_vec, success);

      if (!success)
        stork::runtime_error("Currently, only the dummy ciphertext factory and cleartext results are implemented!");

    }
    return py::cast(result_vec);
  }

  /// Convert the ABC AST to CPP pseudo-code
  std::string to_cpp_string() {
    std::stringstream ss;
    ProgramPrintVisitor v(ss);
    programAst->accept(v);
    return ss.str();
  }
};

PYBIND11_MODULE(_abc_wrapper, m) {
  m.doc() = "Wrapper to export ABC's functionality to read and execute an AST from a JSON file.";

  py::class_<ABCProgramWrapper>(m, "ABCProgramWrapper")
      .def(py::init<const std::string, const std::string>(),
          "Create a program and pre-process the given JSON version of an ABC AST to the CPP version of it."\
          "The second argument is a JSON dump of the arguments and their secret tainting: "\
          "var_name = {\"opt\": Bool, \"value\": value, \"secret\": Bool}"\
          "See _args_to_dict in ABCVisitor.")
      .def("execute", &ABCProgramWrapper::execute,
           "Execute the compiled program on the given inputs and outputs")
      .def("to_cpp_string", &ABCProgramWrapper::to_cpp_string,
           "Convert the ABC AST to CPP pseudo-code");
}
