#include <algorithm>

#include "gtest/gtest.h"

#include "ast_opt/runtime/DummyCiphertext.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/runtime/DummyCiphertextFactory.h"
#include "ast_opt/runtime/Cleartext.h"

class DummyCiphertextFactoryTest : public ::testing::Test {
 protected:
  std::unique_ptr<DummyCiphertextFactory> scf;

  void SetUp() override {
    scf = std::make_unique<DummyCiphertextFactory>();
  }

  void checkCiphertextData(
      AbstractCiphertext &abstractCiphertext,
      const std::vector<int64_t> &expectedValues) {

    // decrypt ciphertext
    std::vector<int64_t> result;
    scf->decryptCiphertext(abstractCiphertext, result); // this should give the data vector

    // check that provided values are in decryption result
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      EXPECT_EQ(expectedValues.at(i), result.at(i));
    }
    // check that all remaining ciphertext slots are filled with last value of given input
    for (auto i = expectedValues.size(); i < result.size(); ++i) {
      ASSERT_EQ(expectedValues.back(), result.at(i));
    }
  }
};

TEST_F(DummyCiphertextFactoryTest, createCiphertext) {
// create ciphertext
  std::vector<int64_t> data = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);
  checkCiphertextData(*ctxt, data);
}

// =======================================
// == "CTXT-CTXT" operations with returned result
// =======================================

TEST_F(DummyCiphertextFactoryTest, add) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->add(*ctxt2);
  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that operands are not changed
  checkCiphertextData(*ctxt1, data1);
  checkCiphertextData(*ctxt2, data2);
}

TEST_F(DummyCiphertextFactoryTest, sub) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->subtract(*ctxt2);
  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that operands are not changed
  checkCiphertextData(*ctxt1, data1);
  checkCiphertextData(*ctxt2, data2);
}

TEST_F(DummyCiphertextFactoryTest, multiply) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->multiply(*ctxt2);
  std::vector<int64_t> expectedData = {0, 3, 2, 4, 50, 189};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that operands are not changed
  checkCiphertextData(*ctxt1, data1);
  checkCiphertextData(*ctxt2, data2);
}

// =======================================
// == CTXT-CTXT in-place operations
// =======================================

TEST_F(DummyCiphertextFactoryTest, addInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  ctxt1->addInplace(*ctxt2);
  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};
  checkCiphertextData(*ctxt1, expectedData);
}

TEST_F(DummyCiphertextFactoryTest, subInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  ctxt1->subtractInplace(*ctxt2);
  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};
  checkCiphertextData(*ctxt1, expectedData);
}

TEST_F(DummyCiphertextFactoryTest, multiplyInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  ctxt1->multiplyInplace(*ctxt2);
  std::vector<int64_t> expectedData = {0, 3, 2, 4, 50, 189};
  checkCiphertextData(*ctxt1, expectedData);
}

// =======================================
// == CTXT-PLAIN operations with returned result
// =======================================

Cleartext<int> createCleartextDummy(const std::vector<int> &literalIntValues) {
  std::vector<std::unique_ptr<AbstractExpression>> result;
  for (const auto &val : literalIntValues) {
    result.emplace_back(std::make_unique<LiteralInt>(val));
  }
  return Cleartext<int>(literalIntValues);
}

TEST_F(DummyCiphertextFactoryTest, addPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextDummy(data2);

  auto ctxtResult = ctxt1->addPlain(operandVector);
  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

TEST_F(DummyCiphertextFactoryTest, subPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextDummy(data2);

  auto ctxtResult = ctxt1->subtractPlain(operandVector);
  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

TEST_F(DummyCiphertextFactoryTest, multiplyPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextDummy(data2);

  auto ctxtResult = ctxt1->multiplyPlain(operandVector);
  std::vector<int64_t> expectedData = {0, 3, 2, 4, 50, 189};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

// =======================================
// == CTXT-PLAIN in-place operations
// =======================================

TEST_F(DummyCiphertextFactoryTest, addPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextDummy(data2);

  ctxt1->addPlainInplace(operandVector);
  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};
  checkCiphertextData(*ctxt1, expectedData);
}

TEST_F(DummyCiphertextFactoryTest, subPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextDummy(data2);

  ctxt1->subtractPlainInplace(operandVector);
  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};
  checkCiphertextData(*ctxt1, expectedData);
}

TEST_F(DummyCiphertextFactoryTest, multiplyPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextDummy(data2);

  ctxt1->multiplyPlainInplace(operandVector);
  std::vector<int64_t> expectedData = {0, 3, 2, 4, 50, 189};
  checkCiphertextData(*ctxt1, expectedData);
}
