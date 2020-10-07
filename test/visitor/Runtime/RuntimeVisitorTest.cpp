#include "include/ast_opt/ast/AbstractNode.h"
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/ast/Literal.h"
#include "include/ast_opt/visitor/Runtime/RuntimeVisitor.h"
#include "include/ast_opt/visitor/Runtime/SealCiphertextFactory.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV
class RuntimeVisitorTest : public ::testing::Test {
 protected:
  std::unique_ptr<SealCiphertextFactory> scf;

  void SetUp() override {
    scf = std::make_unique<SealCiphertextFactory>(4096);
  }
};

TEST_F(RuntimeVisitorTest, testInputOutputAst) { /* NOLINT */
  // Test that checks whether we can pass input, run a very simple instruction, and retrieve the output.

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
      int __input1__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      __input0__ = rotate(__input0__, -4);
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = __input0__;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create a SpecialRuntimeVisitor instance
  SpecialRuntimeVisitor srv(*scf, *astInput);

  // run the program
  astProgram->accept(srv);

  std::unordered_map<std::string, std::vector<int64_t>> expectedOutput;
  std::vector<int64_t> data = {7, 7, 7, 7, 43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
  expectedOutput.emplace("y", data);

  auto output = srv.getOutput(*astOutput);
  for (const auto &[identifier, ciphertext] : output) {
    std::vector<int64_t> decryptedValues;
    scf->decryptCiphertext(*ciphertext, decryptedValues);
    EXPECT_TRUE(expectedOutput.count(identifier) > 0);
    auto expected = expectedOutput.at(identifier);
    for (int i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], decryptedValues[i]);
    }
  }
}

TEST_F(RuntimeVisitorTest, t) { /* NOLINT */
  // Test that shows that retrieving the output modifies the AST.

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
      int __input1__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      __input0__ = rotate(__input0__, -4);
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = __input0__;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create a SpecialRuntimeVisitor instance
  SpecialRuntimeVisitor srv(*scf, *astInput);

  // run the program
  astProgram->accept(srv);

  // Get the output --- this modifies the AST such that consecutive calls to getOutput or printOutput lead to an
  // exception. See the note in RuntimeVisitor.cpp for details on how to fix.
  auto output = srv.getOutput(*astOutput);
  EXPECT_THROW(srv.printOutput(*astOutput), std::out_of_range);
}

TEST_F(RuntimeVisitorTest, programExecution) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      if (N < 5) {
        sum = sum-N;
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);
}

#endif
