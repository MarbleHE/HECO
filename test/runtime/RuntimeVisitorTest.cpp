#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/parser/Parser.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/runtime/SealCiphertextFactory.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV

class RuntimeVisitorTest : public ::testing::Test {
 protected:
  std::unique_ptr<SealCiphertextFactory> scf;
  std::unique_ptr<TypeCheckingVisitor> tcv;

  void SetUp() override {
    scf = std::make_unique<SealCiphertextFactory>(4096);
    tcv = std::make_unique<TypeCheckingVisitor>();
  }

  void registerInputVariable(Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  }

  /// A helper method that takes the result produced by the RuntimeVisitor (result) and a list of expected
  /// (identifier, vector of values) pairs that the program should have returned.
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
      } else if (auto
          cleartextBool = dynamic_cast<Cleartext<bool> *>(cipherClearText.get())) {   // result is a cleartext
        auto cleartextData = cleartextBool->getData();
        // required to convert vector<int> to vector<int64_t>
        plainValues.insert(plainValues.end(), cleartextData.begin(), cleartextData.end());
        EXPECT_EQ(plainValues, expectedResult.at(identifier));
      } else {
        throw std::runtime_error("Could not determine type of result.");
      }
    }
  }
};

TEST_F(RuntimeVisitorTest, testRotateNegative) { /* NOLINT */
  // Test that checks whether we can pass input, run a very simple instruction, and retrieve the output.

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
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

  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // create a SpecialRuntimeVisitor instance
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);

  // run the program
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedOutput;
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
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
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
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
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
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  ASSERT_THROW(srv.executeAst(*astProgram), std::runtime_error);
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionCtxtCtxt) { /* NOLINT */
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
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {1032, 34, 222, 4, 22, 44, 3825, 0, 1, 21};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionCtxtPlaintext) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,  22, 11, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int i = 19;
      secret int result = __input0__ *** i;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
      x = result[3];
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {817, 19, 19, 418, 209, 133};
  expectedResult["x"] = {418};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionPlaintextCtxt) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,  22, 11, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int i = 19;
      secret int result = i *** __input0__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
      x = result[3];
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {817, 19, 19, 418, 209, 133};
  expectedResult["x"] = {418};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionPlaintextPlaintext) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      int __input0__ = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
      int __input1__ = {1, 2, 3, 4, 5, 4, 2, 1, 111, 0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int result = __input1__ > __input0__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, false));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, false));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {0, 0, 0, 0, 1, 0, 0, 0, 1, 0};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}

TEST_F(RuntimeVisitorTest, testBinaryExpressionUnsupportedFhe) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int sum = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      secret int result = sum / sum;
      return sum;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = sum;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create a SpecialRuntimeVisitor instance
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  ASSERT_THROW(srv.executeAst(*astProgram), std::runtime_error);
}

TEST_F(RuntimeVisitorTest, testUnaryExpressionPlaintext) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      bool __input0__ = {0, 0, 1, 1, 0, 0, 0, 0, 1, 1};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int result = !__input0__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::BOOL, false));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {1, 1, 0, 0, 1, 1, 1, 1, 0, 0};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}

TEST_F(RuntimeVisitorTest, testUnaryExpressionUnsupportedFhe) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret bool __input0__ = {0, 0, 1, 1, 0, 0, 0, 0, 1, 1};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = !__input0__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::BOOL, false));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  EXPECT_THROW(srv.executeAst(*astProgram), std::runtime_error);
}

TEST_F(RuntimeVisitorTest, testUnsupportedFunction) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      public int foo() {
        return 0;
      }
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  EXPECT_THROW(srv.executeAst(*astProgram), std::runtime_error);
}

TEST_F(RuntimeVisitorTest, testModSwitch) {
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      __input0__ = modswitch(__input0__ , 1);
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = __input0__;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // create a SpecialRuntimeVisitor instance
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);

  // run the program
  astProgram->accept(srv);

  std::unordered_map<std::string, std::vector<int64_t>> expectedOutput;
  expectedOutput.emplace("y", std::vector<int64_t>({43, 1, 1, 1, 22, 11, 425, 0, 1, 7}));

  auto output = srv.getOutput(*astOutput);

  // compare output with expected output
  assertResult(output, expectedOutput);
}


TEST_F(RuntimeVisitorTest, testRotatePositive) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      __input0__ = rotate(__input0__, 6);
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = __input0__;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // create a SpecialRuntimeVisitor instance
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);

  // run the program
  astProgram->accept(srv);

  std::unordered_map<std::string, std::vector<int64_t>> expectedOutput;
  expectedOutput.emplace("y", std::vector<int64_t>({425, 0, 1, 7, 7, 7, 7, 7, 7}));

  auto output = srv.getOutput(*astOutput);

  // compare output with expected output
  assertResult(output, expectedOutput);
}

TEST_F(RuntimeVisitorTest, testForLoop) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int LIMIT = 10;
      secret int result = 0;
      for (int i = 0; i < LIMIT; i = i + 1) {
        result = result + __input0__;
      }
      return;
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodesList;
  auto astProgram = Parser::parse(std::string(program), createdNodesList);

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";

  auto astOutput = Parser::parse(std::string(outputs));

  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // create a RuntimeVisitor instance
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);

  // run the program
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedOutput;
  expectedOutput.emplace("y", std::vector<int64_t>({430, 10, 10, 10, 220, 110, 4250, 0, 10, 70}));

  auto output = srv.getOutput(*astOutput);

  // compare output with expected output
  assertResult(output, expectedOutput);
}

TEST_F(RuntimeVisitorTest, testFullAssignmentToCiphertext) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int fixedKey = {3, 2, 1, 3, 4, 9, 11, 333, 22, 434, 3430, 2211};
      return;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      result = fixedKey;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["result"] = {3, 2, 1, 3, 4, 9, 11, 333, 22, 434, 3430, 2211};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}

TEST_F(RuntimeVisitorTest, testFullAssignmentToPlaintext) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      double __input0__ = {1.25, 2.22, 4.0, 3.22, 11.0, 41.1, 4.0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      double result = __input0__;
      return;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      r = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::DOUBLE, false));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<double>> expectedResult;
  expectedResult["r"] = {1.25, 2.22, 4, 3.22, 11, 41.1, 4};
  auto result = srv.getOutput(*astOutput);

  EXPECT_EQ(result.size(), 1);
  for (const auto &[identifierStr, abstractValueUPtr] : result) {
    ASSERT_TRUE(expectedResult.count(identifierStr) > 0);
    auto resultCleartext = dynamic_cast<Cleartext<double> *>(abstractValueUPtr.get());
    ASSERT_NE(resultCleartext, nullptr);
    auto resultData = resultCleartext->getData();
    auto expectedData = expectedResult.at(identifierStr);
    ASSERT_EQ(resultData.size(), expectedData.size());
    for (int i = 0; i < expectedData.size(); ++i) {
      ASSERT_EQ(expectedData[i], resultData[i]);
    }
  }
}

#endif
