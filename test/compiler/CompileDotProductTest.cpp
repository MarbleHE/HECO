#include <vector>
#include <stdexcept>
#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/utilities/Scope.h"
#include "ast_opt/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/parser/Parser.h"
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

  // TODO [mh]: fix when it's clear which result is returned by compile

//  /// A helper method that takes the result produced by the RuntimeVisitor (result) and a list of expected
//  /// (identifier, vector of values) pairs that the program should have returned.
//  /// \param result The generated result retrieved by getOutput from the RuntimeVisitor.
//  /// \param expectedResult The expected result that the program should have been produced.
//  auto assertResult = [&scf](const OutputIdentifierValuePairs &result,
//                      const std::unordered_map<std::string, std::vector<int64_t>> &expectedResult) {
//    // Check that the number of results match the number of expected results
//    EXPECT_EQ(result.size(), expectedResult.size());
//
//    for (const auto &[identifier, cipherClearText] : result) {
//      // Check that the result we are currently processing is indeed an expected result
//      EXPECT_EQ(expectedResult.count(identifier), 1);
//
//      // for checking the value, distinguish between a ciphertext (requires decryption) and plaintext
//      std::vector<int64_t> plainValues;
//      if (auto ciphertext = dynamic_cast<AbstractCiphertext *>(cipherClearText.get())) {        // result is a ciphertxt
//        scf->decryptCiphertext(*ciphertext, plainValues);
//        const auto &expResultVec = expectedResult.at(identifier);
//        // to avoid comparing the expanded values (last element of ciphertext is repeated to all remaining slots), we
//        // only compare the values provided in the expectedResult map
//        for (int i = 0; i < expResultVec.size(); ++i) {
//          EXPECT_EQ(plainValues.at(i), expectedResult.at(identifier).at(i));
//        }
//      } else if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(cipherClearText.get())) {   // result is a cleartext
//        auto cleartextData = cleartextInt->getData();
//        // required to convert vector<int> to vector<int64_t>
//        plainValues.insert(plainValues.end(), cleartextData.begin(), cleartextData.end());
//        EXPECT_EQ(plainValues, expectedResult.at(identifier));
//      } else if (auto
//          cleartextBool = dynamic_cast<Cleartext<bool> *>(cipherClearText.get())) {   // result is a cleartext
//        auto cleartextData = cleartextBool->getData();
//        // required to convert vector<int> to vector<int64_t>
//        plainValues.insert(plainValues.end(), cleartextData.begin(), cleartextData.end());
//        EXPECT_EQ(plainValues, expectedResult.at(identifier));
//      } else {
//        throw std::runtime_error("Could not determine type of result.");
//      }
//    }
//  };
//
//  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
//  expectedResult["sum"] = {5205};
//
//  assertResult(result, expectedResult);
}