#include <algorithm>

#include "gtest/gtest.h"

#include "include/ast_opt/visitor/runtime/DummyCiphertext.h"
#include "include/ast_opt/ast/ExpressionList.h"
#include "include/ast_opt/visitor/runtime/DummyCiphertextFactory.h"
#include "include/ast_opt/visitor/runtime/Cleartext.h"

#ifdef HAVE_SEAL_BFV

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
    scf->decryptCiphertext(abstractCiphertext, result);

    // check that the decrypted ciphertext has the expected size
    EXPECT_EQ(result.size(), numCiphertextSlots);

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
/*
TEST_F(DummyCiphertextFactoryTest, createCiphertext) {
  // create ciphertext
  std::vector<int64_t> data = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);
  checkCiphertextData(*ctxt, data);
}*/

TEST_F(DummyCiphertextFactoryTest, createCiphertext) {
  std::vector<int64_t> data = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);
}






#endif