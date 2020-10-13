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

  /// A helper method that takes the result produced by the RuntimeVisitor (result) and a list of expected
  /// (identifier, vector of values) pairs that the program should have been returned.
  /// \param result The generated result retrieved by getOutput from the RuntimeVisitor.
  /// \param expectedResult The expected result that the program should have been produced.
  void assertResult(const OutputIdentifierValuePairs &result,
                    const std::unordered_map<std::string, std::vector<int64_t>> &expectedResult) {
    // Check that the number of results match the number of expected results
    EXPECT_EQ(result.size(), expectedResult.size());

    for (const auto &[identifier, cipherClearText] : result) {
      // Check that the result we are currently processing is indeed an expected result
      EXPECT_EQ(expectedResult.count(identifier), 1);

      // for checking the value, distinguish between a ciphertext (requires decryption) and plaintext
      std::vector<int64_t> plainValues;
      if (auto ciphertext = dynamic_cast<AbstractCiphertext *>(cipherClearText.get())) {        // result is a ciphertxt
        scf->decryptCiphertext(*ciphertext, plainValues);
        const auto &expResultVec = expectedResult.at(identifier);
        // to avoid comparing the expanded values (last element of ciphertext is repeated to all remaining slots), we
        // only compare the values provided in the expectedResult map
        for (int i = 0; i < expResultVec.size(); ++i) {
          EXPECT_EQ(plainValues.at(i), expectedResult.at(identifier).at(i));
        }
      } else if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(cipherClearText.get())) {   // result is a cleartext
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

TEST_F(RuntimeVisitorTest, testSimpleBinaryExpression) { /* NOLINT */
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

TEST_F(RuntimeVisitorTest, testCleartext) { /* NOLINT */
  Cleartext<int> cleartextA(std::vector<int>({2, 3, 4, 1, 1}));
  EXPECT_FALSE(cleartextA.allEqual(1));
  EXPECT_FALSE(cleartextA.allEqual(2));
  EXPECT_FALSE(cleartextA.allEqual(3));
  EXPECT_FALSE(cleartextA.allEqual(4));

  Cleartext<int> cleartextB(std::vector<int>({1}));
  EXPECT_TRUE(cleartextB.allEqual(1));

  Cleartext<int> cleartextC(std::vector<int>({2, 2, 2, 2}));
  EXPECT_TRUE(cleartextC.allEqual(2));
}

TEST_F(RuntimeVisitorTest, testIndexedPlaintextAssignment) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
      int __input1__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int sum = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      sum[3] = 333;
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

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {1, 2, 3, 333, 5, 6, 7, 8, 9, 10};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}

TEST_F(RuntimeVisitorTest, testIndexedCiphertextAssignment) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
      int __input1__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int sum = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      sum[3] = 333;
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
  ASSERT_THROW(srv.executeAst(*astProgram), std::runtime_error);
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionCtxtCtxt) { /* NOLINT */  // FIXME: Make this test work
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = __input0__ *** __input1__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  TypeCheckingVisitor typeCheckingVisitor;
  auto rootScope = std::make_unique<Scope>(*astProgram);
  auto input0 = std::make_unique<ScopedIdentifier>(*rootScope, "__input0__");
  auto input1 = std::make_unique<ScopedIdentifier>(*rootScope, "__input1__");
  typeCheckingVisitor.addVariableDatatype(*input0, Datatype(Type::INT, true));
  typeCheckingVisitor.addVariableDatatype(*input1, Datatype(Type::INT, true));
  rootScope->addIdentifier("__input0__");
  rootScope->addIdentifier("__input1__");
  typeCheckingVisitor.setRootScope(std::move(rootScope));
  astProgram->accept(typeCheckingVisitor);

  // run the program and get its output
  auto map = typeCheckingVisitor.getSecretTaintedNodes();
  auto secondAssignmentNode = astInput->begin()->begin();
  std::advance(secondAssignmentNode, 1);
  map.insert_or_assign(astInput->begin()->begin()->getUniqueNodeId(), true);
  map.insert_or_assign(secondAssignmentNode->getUniqueNodeId(), true);

  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {1032, 34, 222, 4, 22, 44, 3825, 0, 1, 21};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionCtxtPlaintext) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionPlaintextCtxt) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionPlaintextPlaintext) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionUnsupportedFhe) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testUnaryExpressionPlaintext) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testUnaryExpressionUnsupportedFhe) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testRotate) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testExpressionListPlaintext) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testForLoop) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testUnsupportedFunction) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testFullAssignmentToCiphertext) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testFullAssignmentToPlaintext) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testVariableDeclarationPlaintext) { /* NOLINT */
  // TODO: Implement me!
}

TEST_F(RuntimeVisitorTest, testVariableDeclarationCiphertext) { /* NOLINT */
  // TODO: Implement me!
}

#endif
