#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "abc/ast_parser/Parser.h"
#include "abc/ast_parser/Errors.h"
#include "abc/ast_utilities/ProgramPrintVisitor.h"
#include "abc/IR/ABC/ABCDialect.h"
#include "abc/ast_utilities/abc_ast_to_mlir_visitor.h"

namespace py = pybind11;
using json = nlohmann::json;

void abc_ast_to_mlir(std::unique_ptr<AbstractNode> programAst) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<abc::ABCDialect>();

  AbcAstToMlirVisitor v(ctx);
  programAst->accept(v);

  auto module = v.getModule();

  // TODO (Miro): Rewrite the Python part to parse the entire content of the ABCContext with block, including the main
  //  function (currently, this would break the exisiting ABC AST execution).
  //  The following returns an empty module, since only functions are added to the module, but the
  //  main function is not parsed atm.
  module.dump();
}

class AbstractValue {
  virtual ~AbstractValue() = default;
};

template<typename T>
class Cleartext : public AbstractValue {
  ~Cleartext() override = default;
};

class AbstractCiphertext {};

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

  static void extract_ciphertext_result_vec(AbstractValue *resultCiphertext,
                                            std::vector<py::object> &result_vec, bool &success) {

    if (auto ciphertext = dynamic_cast<AbstractCiphertext *>(resultCiphertext)) {
      // TODO: we only support dummy ciphertext values. Currently, we cannot pass encrypted data to Python.
      //  do key management and/or export the ciphertext to Python.
      // TODO: there's only support for int64_t values in the dummy ciphertext factory
      std::vector<int64_t> result;
      //TODO: How to actually decryptCiphertext(*ciphertext, result);

      result_vec.push_back(py::cast(result));
      success = true;
    } else {
      success = false;
    }
  }

 public:
  /// Create a program and pre-process the given JSON to a proper ABC AST.
  /// \param program JSON version of ABC AST
  explicit ABCProgramWrapper(const std::string program, const std::string args) {
    programAst = Parser::parseJson(program);

    // Translate ABC AST to MLIR
    abc_ast_to_mlir(programAst->clone());

    // TODO: where can we mark arguments as secret? The args argument has a boolen for those that should be secret, but do
    //    we have to already mark them while/before parsing the AST or only when executing?
    programArgs = json::parse(args);

    // TODO: translate the entire function to mlir instead of separating arguments and program. The new compiler
    //  will no longer require this.
  }

  /// Execute the compiled program on the given inputs and outputs
  // XXX: This assumes the result is in clear text.
  // We use python types as result values to be able to return different types without having to use templates,
  // since binding templates would require us to instantiate different programs depending on their return type.
  py::tuple execute(std::string inputs, std::vector<std::string> outputIdentifiers) {
    // TODO: execute MLIR once the other levels are ready

    // Cloning the programAst is necessary to be able to execute the same program multiple times (which uses the same
    // program AST)

    //TODO: Now that runtime visitor is removed, we need a new way to actually execute
    //auto result = Compiler::compileJson(programAst->clone(), inputs, outputIdentifiers);
    std::vector<py::object> result_vec;

    //for (const auto &[identifier, cipherClearText] : result) {
    //  bool success = false;
    //  auto ciphertextValue = cipherClearText.get();
    //
    //  // Plaintext values
    //  extract_literal_result_vec<bool>(ciphertextValue, result_vec, success);
    //  extract_literal_result_vec<int>(ciphertextValue, result_vec, success);
    //  extract_literal_result_vec<float>(ciphertextValue, result_vec, success);
    //  extract_literal_result_vec<double>(ciphertextValue, result_vec, success);
    //  extract_literal_result_vec<char>(ciphertextValue, result_vec, success);
    //  extract_literal_result_vec<std::string>(ciphertextValue, result_vec, success);
    //
    //  // Ciphertext value
    //  extract_ciphertext_result_vec(ciphertextValue, result_vec, success);
    //
    //  if (!success)
    //    throw stork::runtime_error("Currently, only the dummy ciphertext factory and cleartext results are implemented!");
    //}
    return py::cast(result_vec);
  }

  // TODO: remove once the switch to MLIR is done
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
