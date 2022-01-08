#include <vector>
#include <stdexcept>
#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/ast_utilities/Scope.h"
#include "ast_opt/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/ast_parser/Parser.h"
#include "ast_opt/compiler/Compiler.h"
#include "gtest/gtest.h"

TEST(DotProductTest, pseudoCppCompile) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      int x = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      int y = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      int size = 10;
    )"""";

  // program specification
  const char *program = R""""(
      int sum = 0;
      for (int i = 0; i < size; i = i + 1) {
          sum = sum + x[i]*y[i];
      }
      return sum;
    )"""";

  const std::vector<std::string> outputIdentifiers = {"sum"};

  auto result = Compiler::compile(program, inputs, outputIdentifiers);

  std::unordered_map<std::string, int> expectedResult;
  expectedResult["sum"] = 5205;

  for (const auto &[identifier, cipherClearText] : result) {
    if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(cipherClearText.get())) {   // result is a cleartext
      auto cleartextData = cleartextInt->getData();
      ASSERT_EQ(cleartextData[0], expectedResult[identifier]);
    }
  }
}