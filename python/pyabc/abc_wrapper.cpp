#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "heco/ast_parser/Parser.h"
#include "heco/ast_parser/Errors.h"
#include "heco/ast_utilities/ProgramPrintVisitor.h"
#include "heco/IR/AST/ASTDialect.h"
#include "heco/ast_utilities/AbcAstToMlirVisitor.h"

namespace py = pybind11;
using json = nlohmann::json;

class AbstractValue
{
  virtual ~AbstractValue() = default;
};

template <typename T>
class Cleartext : public AbstractValue
{
  ~Cleartext() override = default;
};

class AbstractCiphertext
{
};

/// Class storing a program and intermediate state. E.g., it stores a pre-compiled program AST to avoid
/// having to pass a unique pointer to Python.
class ABCProgramWrapper
{
private:
  std::unique_ptr<AbstractNode> programAst;
  mlir::ModuleOp module;

  // Those two are only attributes because otherwise the module will not be preserves, since it relies on the
  // context variable to still exist.
  mlir::MLIRContext ctx;
  mlir::OpBuilder *builder;

  // TODO: adapt for extracting results from MLIR or remove entirely
  //  /// Helper function to extract literal (array) and push it to the results vector
  //  template<class T>
  //  static void extract_literal_result_vec(AbstractValue *cipherClearText,
  //                                         std::vector<py::object> &result_vec, bool &success) {
  //
  //    if (auto cleartextInt = dynamic_cast<Cleartext<T> *>(cipherClearText)) {   // result is a cleartext
  //      success = true;
  //      auto cleartextData = py::cast(cleartextInt->getData());
  //      result_vec.push_back(cleartextData);
  //    }
  //  }
  //
  //  static void extract_ciphertext_result_vec(AbstractValue *resultCiphertext,
  //                                            std::vector<py::object> &result_vec, bool &success) {
  //
  //    if (auto ciphertext = dynamic_cast<AbstractCiphertext *>(resultCiphertext)) {
  //      // TODO: we only support dummy ciphertext values. Currently, we cannot pass encrypted data to Python.
  //      //  do key management and/or export the ciphertext to Python.
  //      // TODO: there's only support for int64_t values in the dummy ciphertext factory
  //      std::vector<int64_t> result;
  //      //TODO: How to actually decryptCiphertext(*ciphertext, result);
  //
  //      result_vec.push_back(py::cast(result));
  //      success = true;
  //    } else {
  //      success = false;
  //    }
  //  }

public:
  /// Create a program and pre-process the given JSON to a proper ABC AST.
  /// \param program JSON version of ABC AST
  explicit ABCProgramWrapper(const std::string program)
  {
    programAst = Parser::parseJson(program);

    ctx.getOrLoadDialect<heco::ABCDialect>();
    builder = new mlir::OpBuilder(&ctx);
    module = mlir::ModuleOp::create(builder->getUnknownLoc());

    // Translate ABC AST to MLIR
    AbcAstToMlirVisitor v(ctx);
    programAst->clone()->accept(v);

    module.getRegion().push_back(v.getBlockPtr());
  }

  void add_fn(std::string fn)
  {
    auto fn_parsed = Parser::parseJson(fn);

    // Translate ABC AST of fn to MLIR
    AbcAstToMlirVisitor v(ctx);
    fn_parsed->accept(v);

    module.getRegion().push_back(v.getBlockPtr());
  }

  /// Execute the compiled program on the given inputs and outputs
  // XXX: This assumes the result is in clear text.
  // We use python types as result values to be able to return different types without having to use templates,
  // since binding templates would require us to instantiate different programs depending on their return type.
  py::tuple execute(std::string inputs, std::vector<std::string> outputIdentifiers)
  {
    // TODO: execute MLIR once the other levels are ready

    // Cloning the programAst is necessary to be able to execute the same program multiple times (which uses the same
    // program AST)

    // TODO: Now that runtime visitor is removed, we need a new way to actually execute
    // auto result = Compiler::compileJson(programAst->clone(), inputs, outputIdentifiers);
    std::vector<py::object> result_vec;

    // for (const auto &[identifier, cipherClearText] : result) {
    //   bool success = false;
    //   auto ciphertextValue = cipherClearText.get();
    //
    //   // Plaintext values
    //   extract_literal_result_vec<bool>(ciphertextValue, result_vec, success);
    //   extract_literal_result_vec<int>(ciphertextValue, result_vec, success);
    //   extract_literal_result_vec<float>(ciphertextValue, result_vec, success);
    //   extract_literal_result_vec<double>(ciphertextValue, result_vec, success);
    //   extract_literal_result_vec<char>(ciphertextValue, result_vec, success);
    //   extract_literal_result_vec<std::string>(ciphertextValue, result_vec, success);
    //
    //   // Ciphertext value
    //   extract_ciphertext_result_vec(ciphertextValue, result_vec, success);
    //
    //   if (!success)
    //     throw stork::runtime_error("Currently, only the dummy ciphertext factory and cleartext results are implemented!");
    // }
    return py::cast(result_vec);
  }

  void dump()
  {
    module.dump();
  }
};

PYBIND11_MODULE(_abc_wrapper, m)
{
  m.doc() = "Wrapper to export ABC's functionality to read and execute an AST from a JSON file.";

  py::class_<ABCProgramWrapper>(m, "ABCProgramWrapper")
      .def(py::init<const std::string>(),
           "Create a program and pre-process the given JSON version of an ABC AST to the CPP version of it.")
      .def("add_fn", &ABCProgramWrapper::add_fn,
           "Pre-process a json string of a function and add it.")
      .def("execute", &ABCProgramWrapper::execute,
           "Execute the compiled program on the given inputs and outputs")
      .def("dump", &ABCProgramWrapper::dump,
           "Print the parsed MLIR");
}
