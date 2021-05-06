#include <algorithm>

#include "gtest/gtest.h"

#include "include/ast_opt/visitor/runtime/SealCiphertext.h"
#include "include/ast_opt/ast/ExpressionList.h"
#include "include/ast_opt/visitor/runtime/SealCiphertextFactory.h"
#include "include/ast_opt/visitor/runtime/Cleartext.h"

#ifdef HAVE_SEAL_BFV

class SealCiphertextFactoryTest : public ::testing::Test {
 protected:
  const int numCiphertextSlots = 4096;

  std::unique_ptr<SealCiphertextFactory> scf;

  void SetUp() override {
    scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
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

TEST_F(SealCiphertextFactoryTest, createCiphertext) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);
  checkCiphertextData(*ctxt, data);
}

TEST_F(SealCiphertextFactoryTest, rotateCiphertextLhs) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data = {123456, 3, 1, 4, 5, 9, 5, 2, 1, 5};
  auto const initialSizeData = data.size();
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);
  checkCiphertextData(*ctxt, data);

  auto const steps = 4;

  auto rotatedCtxt = ctxt->rotateRows(steps);
  std::vector<int64_t> dv;
  scf->decryptCiphertext(*rotatedCtxt, dv);

  // check that original ciphertext is unchanged
  checkCiphertextData(*ctxt, data);

  // check rotated ciphertext
  auto const nextRowStartIdx = scf->getCiphertextSlotSize()/2;  // see the SEAL docs (Evaluator::rotate_rows)
  for (int i = 0; i < dv.size(); ++i) {
    if (i < std::min<int>(initialSizeData - steps, nextRowStartIdx - steps)) {
      // compare values that moved to the beginning of the ciphertext
      EXPECT_EQ(data.at(i + steps), dv.at(i));
    } else if (i >= nextRowStartIdx - steps && i < nextRowStartIdx) {
      // compare values that "wrapped" due to cyclicality of the rotation
      EXPECT_EQ(data.at(i - (nextRowStartIdx - steps)), dv.at(i));
    } else {
      // compare any other values (the ones that are filled with the last element of the input data in the original
      // ciphertext)
      EXPECT_EQ(data.at(initialSizeData - 1), dv.at(i));
    }
  }
}

TEST_F(SealCiphertextFactoryTest, rotateCiphertextRhs) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data = {123456, 3, 1, 4, 5, 9, 5, 2, 1, 5};
  auto const initialSizeData = data.size();
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);
  checkCiphertextData(*ctxt, data);

  auto const steps = -24;

  auto rotatedCtxt = ctxt->rotateRows(steps);
  std::vector<int64_t> dv;
  scf->decryptCiphertext(*rotatedCtxt, dv);

  // check that original ciphertext is unchanged
  checkCiphertextData(*ctxt, data);

  // check rotated ciphertext
  auto const nextRowStartIdx = scf->getCiphertextSlotSize()/2;  // see the SEAL docs (Evaluator::rotate_rows)
  for (int i = 0; i < dv.size(); ++i) {
    if (i < abs(steps) || i >= abs(steps) + initialSizeData) {
      // compare filled values
      EXPECT_EQ(data.at(initialSizeData - 1), dv.at(i));
    } else {
      // compare moved values
      EXPECT_EQ(data.at(i + steps), dv.at(i));
    }
  }
}

TEST_F(SealCiphertextFactoryTest, rotateCiphertextInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data = {123456, 3, 1, 4, 5, 9, 5, 2, 1, 5};
  auto const initialSizeData = data.size();
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);

  auto const steps = 4;

  ctxt->rotateRowsInplace(steps);
  std::vector<int64_t> dv;
  scf->decryptCiphertext(*ctxt, dv);

  size_t j = 0;
  auto const nextRowStartIdx = scf->getCiphertextSlotSize()/2;  // see the SEAL docs (Evaluator::rotate_rows)
  for (int i = 0; i < dv.size(); ++i, ++j) {
    if (i < std::min<int>(initialSizeData - steps, nextRowStartIdx - steps)) {
      // compare values that moved to the beginning of the ciphertext
      EXPECT_EQ(data.at(i + steps), dv.at(i));
    } else if (i >= nextRowStartIdx - steps && i < nextRowStartIdx) {
      // compare values that "wrapped" due to cyclicality of the rotation
      EXPECT_EQ(data.at(i - (nextRowStartIdx - steps)), dv.at(i));
    } else {
      // compare any other values (the ones that are filled with the last element of the input data in the original
      // ciphertext)
      EXPECT_EQ(data.at(initialSizeData - 1), dv.at(i));
    }
  }
}

// =======================================
// == CTXT-CTXT operations with returned result
// =======================================

TEST_F(SealCiphertextFactoryTest, add) { /* NOLINT */
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

TEST_F(SealCiphertextFactoryTest, sub) { /* NOLINT */
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

TEST_F(SealCiphertextFactoryTest, multiply) { /* NOLINT */
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

TEST_F(SealCiphertextFactoryTest, addInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  ctxt1->addInplace(*ctxt2);
  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};
  checkCiphertextData(*ctxt1, expectedData);
}

TEST_F(SealCiphertextFactoryTest, subInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  ctxt1->subtractInplace(*ctxt2);
  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};
  checkCiphertextData(*ctxt1, expectedData);
}

TEST_F(SealCiphertextFactoryTest, multiplyInplace) { /* NOLINT */
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

Cleartext<int> createCleartext(const std::vector<int> &literalIntValues) {
  std::vector<std::unique_ptr<AbstractExpression>> result;
  for (const auto &val : literalIntValues) {
    result.emplace_back(std::make_unique<LiteralInt>(val));
  }
  return Cleartext<int>(literalIntValues);
}

TEST_F(SealCiphertextFactoryTest, addPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartext(data2);

  auto ctxtResult = ctxt1->addPlain(operandVector);
  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

TEST_F(SealCiphertextFactoryTest, subPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartext(data2);

  auto ctxtResult = ctxt1->subtractPlain(operandVector);
  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

TEST_F(SealCiphertextFactoryTest, multiplyPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartext(data2);

  auto ctxtResult = ctxt1->multiplyPlain(operandVector);
  std::vector<int64_t> expectedData = {0, 3, 2, 4, 50, 189};
  checkCiphertextData(*ctxtResult, expectedData);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

// =======================================
// == CTXT-PLAIN in-place operations
// =======================================

TEST_F(SealCiphertextFactoryTest, addPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartext(data2);

  ctxt1->addPlainInplace(operandVector);
  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};
  checkCiphertextData(*ctxt1, expectedData);
}

TEST_F(SealCiphertextFactoryTest, subPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartext(data2);

  ctxt1->subtractPlainInplace(operandVector);
  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};
  checkCiphertextData(*ctxt1, expectedData);
}

TEST_F(SealCiphertextFactoryTest, multiplyPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartext(data2);

  ctxt1->multiplyPlainInplace(operandVector);
  std::vector<int64_t> expectedData = {0, 3, 2, 4, 50, 189};
  checkCiphertextData(*ctxt1, expectedData);
}

#endif


