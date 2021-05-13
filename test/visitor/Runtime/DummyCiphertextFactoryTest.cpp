#include <algorithm>

#include "gtest/gtest.h"

#include "include/ast_opt/visitor/runtime/DummyCiphertext.h"
#include "include/ast_opt/ast/ExpressionList.h"
#include "include/ast_opt/visitor/runtime/DummyCiphertextFactory.h"
#include "include/ast_opt/visitor/runtime/Cleartext.h"

class DummyCiphertextFactoryTest : public ::testing::Test {
 protected:
  const int numCiphertextSlots = 4096;

  std::unique_ptr<DummyCiphertextFactory> scf;

  void SetUp() override {
    scf = std::make_unique<DummyCiphertextFactory>(numCiphertextSlots);
  }

  void checkCiphertextData(
      AbstractCiphertext &abstractCiphertext,
      const std::vector<int64_t> &expectedValues) {

    // decrypt ciphertext
    std::vector<int64_t> result;
    scf->decryptCiphertext(abstractCiphertext, result); // this should give the data vector

    // check that provided values are in decryption result
    for (int i = 0; i < expectedValues.size(); ++i) {
      EXPECT_EQ(expectedValues.at(i), result.at(i));
    }
    // check that all remaining ciphertext slots are filled with last value of given input
    for (int i = expectedValues.size(); i < result.size(); ++i) {
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

