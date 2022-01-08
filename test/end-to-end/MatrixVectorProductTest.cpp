#include <vector>
#include <stdexcept>
#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/ast_utilities/Scope.h"
#include "ast_opt/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/ast_parser/Parser.h"
#include "gtest/gtest.h"

/// Original, plain C++ program for a matrix-vector product
///
/// \param matrix matrix of size m x n (row-major)
/// \param vec vector of size n
/// \return matrix * vec
std::vector<int> matrixVectorProduct(const std::vector<std::vector<int>> &matrix, const std::vector<int> &vec) {

  if (matrix.empty() || matrix[0].size()!=vec.size())
    throw std::runtime_error("Vectors  in dot product must have the same length.");

  size_t m = matrix.size();
  size_t n = vec.size();

  std::vector<int> result(matrix.size());
  for (size_t i = 0; i < m; ++i) {
    int sum = 0;
    for (size_t j = 0; j < n; ++j) {
      sum += matrix[i][j]*vec[j];
    }
    result[i] = sum;
  }
  return result;
}

TEST(MatrixVectorProduct, clearTextEvaluation) { /* NOLINT */
// program's input
  const char *inputs = R""""(
      int matrix = {1,  1,   1,  1,  1, 1, 1,  1, 1};
      int vec = {24, 34, 222};
      int n = 3;
      int m = 3;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

// program specification
  const char *program = R""""(
      int result = {0,0,0};
      for (int i = 0; i < m; i = i  + 1) {
        int sum = 0;
        for (int j = 0; j < n; j = j + 1) {
          sum = sum + matrix[i*m+j]*vec[j];
         }
        result[i] = sum;
      }
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      result = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  auto scf = std::make_unique<DummyCiphertextFactory>();
  auto tcv = std::make_unique<TypeCheckingVisitor>();

  // create and prepopulate TypeCheckingVisitor
  auto registerInputVariable = [&tcv](Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  };

  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "matrix", Datatype(Type::INT, false));
  registerInputVariable(*rootScope, "vec", Datatype(Type::INT, false));
  registerInputVariable(*rootScope, "m", Datatype(Type::INT, false));
  registerInputVariable(*rootScope, "n", Datatype(Type::INT, false));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  //TODO: Change it so that by passing in an empty secretTaintingMap, we can get the RuntimeVisitor to execute everything "in the clear"!
  auto empty = std::unordered_map<std::string, bool>();
  RuntimeVisitor srv(*scf, *astInput, empty);
  srv.executeAst(*astProgram);


  /// A helper method that takes the result produced by the RuntimeVisitor (result) and a list of expected
  /// (identifier, vector of values) pairs that the program should have returned.
  /// \param result The generated result retrieved by getOutput from the RuntimeVisitor.
  /// \param expectedResult The expected result that the program should have been produced.
  auto assertResult = [&scf](const OutputIdentifierValuePairs &result,
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
  };

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["result"] = {280, 280, 280};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}