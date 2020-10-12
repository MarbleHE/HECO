#include "include/ast_opt/ast/AbstractNode.h"
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/ast/Literal.h"
#include "include/ast_opt/visitor/runtime/RuntimeVisitor.h"
#include "include/ast_opt/visitor/runtime/SealCiphertextFactory.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV

class RuntimeVisitorTest : public ::testing::Test {
 protected:
  std::unique_ptr<SealCiphertextFactory> scf;

  void SetUp() override {
    scf = std::make_unique<SealCiphertextFactory>(4096);
  }

  void assertResult(const OutputIdentifierValuePairs &result,
                    const std::unordered_map<std::string, std::vector<int64_t>> &expectedResult) {
    EXPECT_EQ(result.size(), expectedResult.size());

    for (const auto &[identifier, cipherClearText] : result) {
      EXPECT_EQ(expectedResult.count(identifier), 1);

      std::vector<int64_t> plainValues;
      if (auto ciphertext = dynamic_cast<AbstractCiphertext *>(cipherClearText.get())) {
        scf->decryptCiphertext(*ciphertext, plainValues);
        const auto &expResultVec = expectedResult.at(identifier);
        // to avoid comparing the expanded values (last element of ciphertext is repeated to all remaining slots), we
        // only compare the values provided in the expectedResult map
        for (int i = 0; i < expResultVec.size(); ++i) {
          EXPECT_EQ(plainValues.at(i), expectedResult.at(identifier).at(i));
        }
      } else if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(cipherClearText.get())) {
        auto cleartextData = cleartextInt->getData();
        // required to convert vector<int> to vector<int64_t>
        plainValues.insert(plainValues.end(), cleartextData.begin(), cleartextData.end());
        EXPECT_EQ(plainValues, expectedResult.at(identifier));
      } else {
        throw std::runtime_error("Could not determine type of result.");
      }
    }
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
  SecretTaintedNodesMap secretTaintedNodesMap;
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);

  // run the program
  astProgram->accept(srv);

  std::unordered_map<std::string, std::vector<int64_t>> expectedOutput;
//  std::vector<int64_t> data = {7, 7, 7, 7, 43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
  expectedOutput.emplace("y", std::vector<int64_t>({7, 7, 7, 7, 43, 1, 1, 1, 22, 11, 425, 0, 1, 7}));

  auto output = srv.getOutput(*astOutput);

  // compare output with expected output
  assertResult(output, expectedOutput);
}

TEST_F(RuntimeVisitorTest, testSimpleBinaryExpression) {  /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
      int __input1__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int sum = 10+25;
      return sum;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = sum;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create a SpecialRuntimeVisitor instance
  TypeCheckingVisitor typeCheckingVisitor;
  astProgram->accept(typeCheckingVisitor);

  // run the program and get its output
  auto map = typeCheckingVisitor.getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);
  auto result = srv.getOutput(*astOutput);

  // define expected output
  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {35};

  // compare output with expected output
  assertResult(result, expectedResult);
}

#endif
