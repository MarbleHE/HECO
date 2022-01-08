#include <vector>
#include <stdexcept>
#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/ast_utilities/Scope.h"
#include "ast_opt/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/ast_parser/Parser.h"
#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV
#include "bench/HammingDistance.h"
#endif

/// Original, plain C++ program for the hamming distance between two vectors
///
/// \param x vector of size n
/// \param y vector of size n
/// \return hamming distance between the two vectors
int hammingDistance(const std::vector<bool> &x, const std::vector<bool> &y) {

  if (x.size()!=y.size()) throw std::runtime_error("Vectors  in hamming distance must have the same length.");
  int sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sum += x[i]!=y[i];
  }
  return sum;
}

#ifdef HAVE_SEAL_BFV
TEST(HammingDistanceTest, NaiveClear_Porcupine_Equivalence) { /* NOLINT */
  // Create two vectors of bits (booleans),
  // TODO: Create test values from fixed random seed
  std::vector<bool> a(4, 0);
  std::vector<bool> b(4, 1);

  MultiTimer dummy = MultiTimer();
  auto result = encryptedHammingDistancePorcupine(dummy, a, b, 2 << 13);

  // Compare to reference cleartext implementation
  EXPECT_EQ(hammingDistance(a, b), result);
}

TEST(HammingDistanceTest, NaiveClear_Batched_Equivalence) { /* NOLINT */
  // Create two vectors of bits (booleans),
  // TODO: Create test values from fixed random seed
  std::vector<bool> a(4, 0);
  std::vector<bool> b(4, 1);

  MultiTimer dummy = MultiTimer();
  auto result = encryptedBatchedHammingDistance(dummy, a, b, 2 << 13);

  // Compare to reference cleartext implementation
  EXPECT_EQ(hammingDistance(a, b), result);
}

#endif //HAVE_SEAL_BFV

//Note: Hamming distance over binary vectors can be computed efficiently in Z_p by using NEQ = XOR = (a-b)^2

TEST(HammingDistanceTest, clearTextEvaluation) { /* NOLINT */
  /// program's input
  const char *inputs = R""""(
      int x = {1,1,0,1};
      int y = {1,0,1,1};
      int n = 4;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));


  /// program specification
  const char *program = R""""(
    int sum = 0;
    for (int i = 0; i < n; i = i + 1) {
      sum = sum + (x[i]-y[i])*(x[i]-y[i]);
    }
    return sum;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

// program's output
  const char *outputs = R""""(
      sum = sum;
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
  registerInputVariable(*rootScope, "x", Datatype(Type::INT, false));
  registerInputVariable(*rootScope, "y", Datatype(Type::INT, false));
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
  expectedResult["sum"] = {2};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}
