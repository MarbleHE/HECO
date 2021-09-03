#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ast_opt/compiler/Compiler.h"
#include "ast_opt/runtime/Cleartext.h"

namespace py = pybind11;

// TODO [mh]: make this general for non-int return types. Also, this assumes the result is in clear text.
std::vector<int> runIntProgram(std::string program, std::string inputs, std::vector<std::string> outputIdentifiers) {
  auto result = Compiler::compileJson(program, inputs, outputIdentifiers);
  std::vector<int> result_vec;

  for (const auto &[identifier, cipherClearText] : result) {
    if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(cipherClearText.get())) {   // result is a cleartext
      auto cleartextData = cleartextInt->getData();
      result_vec.push_back(cleartextData[0]);
    }
  }

  return result_vec;
}

PYBIND11_MODULE(_abc_wrapper, m) {
  m.doc() = "Wrapper to export ABC's functionality to read and execute an AST from a JSON file.";

  m.def("runIntProgram", &runIntProgram, "read and execute an AST with an integer return value from a JSON file");
}
