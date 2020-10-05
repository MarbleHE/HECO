#include "include/ast_opt/visitor/Runtime/SealCiphertext.h"
#include "include/ast_opt/visitor/Runtime/SealCiphertextFactory.h"
#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV

TEST(SealCiphertextFactoryTest, createCiphertext) { /* NOLINT */
  // create ciphertext
  const auto numCiphertextSlots = 4096;
  std::unique_ptr<SealCiphertextFactory> scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
  std::vector<int64_t> data = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);

  // decrypt ciphertext
  std::vector<int64_t> result;
  scf->decryptCiphertext(*ctxt, result);

  EXPECT_EQ(result.size(), numCiphertextSlots);

  // check that provided values are in decryption result
  for (int i = 0; i < data.size(); ++i) {
    EXPECT_EQ(data.at(i), result.at(i));
  }
  // check that all remaining ciphertext slots are filled with last value of given input
  for (int i = data.size(); i < result.size(); ++i) {
    ASSERT_EQ(data.back(), result.at(i));
  }
}

TEST(SealCiphertextFactoryTest, multiply) { /* NOLINT */
  // create ciphertexts
  const auto numCiphertextSlots = 4096;
  std::unique_ptr<SealCiphertextFactory> scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->multiply(*ctxt2);

  // decrypt ciphertext
  std::vector<int64_t> result;
  scf->decryptCiphertext(*ctxtResult, result);

  std::vector<int64_t> expectedData = {0, 3, 2, 4, 50, 189};

  EXPECT_EQ(result.size(), numCiphertextSlots);

  // check that provided values are in decryption result
  for (int i = 0; i < expectedData.size(); ++i) {
    EXPECT_EQ(expectedData.at(i), result.at(i));
  }
  // check all remaining ciphertext slots that contain result of (data1.last * data2.last)
  for (int i = expectedData.size(); i < result.size(); ++i) {
    ASSERT_EQ(data1.back()*data2.back(), result.at(i));
  }
}

TEST(SealCiphertextFactoryTest, multiplyInplace) { /* NOLINT */
  // create ciphertexts
  const auto numCiphertextSlots = 4096;
  std::unique_ptr<SealCiphertextFactory> scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  ctxt1->multiplyInplace(*ctxt2);

  // decrypt ciphertext
  std::vector<int64_t> result;
  scf->decryptCiphertext(*ctxt1, result);

  std::vector<int64_t> expectedData = {0, 3, 2, 4, 50, 189};

  EXPECT_EQ(result.size(), numCiphertextSlots);

  // check that provided values are in decryption result
  for (int i = 0; i < expectedData.size(); ++i) {
    EXPECT_EQ(expectedData.at(i), result.at(i));
  }
  // check all remaining ciphertext slots that contain result of (data1.last * data2.last)
  for (int i = expectedData.size(); i < result.size(); ++i) {
    ASSERT_EQ(data1.back()*data2.back(), result.at(i));
  }
}

TEST(SealCiphertextFactoryTest, add) { /* NOLINT */
  // create ciphertexts
  const auto numCiphertextSlots = 4096;
  std::unique_ptr<SealCiphertextFactory> scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->add(*ctxt2);

  // decrypt ciphertext
  std::vector<int64_t> result;
  scf->decryptCiphertext(*ctxtResult, result);

  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};

  EXPECT_EQ(result.size(), numCiphertextSlots);

  // check that provided values are in decryption result
  for (int i = 0; i < expectedData.size(); ++i) {
    EXPECT_EQ(expectedData.at(i), result.at(i));
  }
  // check all remaining ciphertext slots that contain result of (data1.last * data2.last)
  for (int i = expectedData.size(); i < result.size(); ++i) {
    ASSERT_EQ(data1.back() + data2.back(), result.at(i));
  }
}

TEST(SealCiphertextFactoryTest, addInplace) { /* NOLINT */
  // create ciphertexts
  const auto numCiphertextSlots = 4096;
  std::unique_ptr<SealCiphertextFactory> scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  ctxt1->addInplace(*ctxt2);

  // decrypt ciphertext
  std::vector<int64_t> result;
  scf->decryptCiphertext(*ctxt1, result);

  std::vector<int64_t> expectedData = {3, 4, 3, 5, 15, 30};

  EXPECT_EQ(result.size(), numCiphertextSlots);

  // check that provided values are in decryption result
  for (int i = 0; i < expectedData.size(); ++i) {
    EXPECT_EQ(expectedData.at(i), result.at(i));
  }
  // check all remaining ciphertext slots that contain result of (data1.last * data2.last)
  for (int i = expectedData.size(); i < result.size(); ++i) {
    ASSERT_EQ(data1.back() + data2.back(), result.at(i));
  }
}

TEST(SealCiphertextFactoryTest, sub) { /* NOLINT */
  // create ciphertexts
  const auto numCiphertextSlots = 4096;
  std::unique_ptr<SealCiphertextFactory> scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->subtract(*ctxt2);

  // decrypt ciphertext
  std::vector<int64_t> result;
  scf->decryptCiphertext(*ctxtResult, result);

  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};

  EXPECT_EQ(result.size(), numCiphertextSlots);

  // check that provided values are in decryption result
  for (int i = 0; i < expectedData.size(); ++i) {
    EXPECT_EQ(expectedData.at(i), result.at(i));
  }
  // check all remaining ciphertext slots that contain result of (data1.last * data2.last)
  for (int i = expectedData.size(); i < result.size(); ++i) {
    ASSERT_EQ(data1.back() - data2.back(), result.at(i));
  }
}

TEST(SealCiphertextFactoryTest, subInplace) { /* NOLINT */
  // create ciphertexts
  const auto numCiphertextSlots = 4096;
  std::unique_ptr<SealCiphertextFactory> scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  ctxt1->subtractInplace(*ctxt2);

  // decrypt ciphertext
  std::vector<int64_t> result;
  scf->decryptCiphertext(*ctxt1, result);

  std::vector<int64_t> expectedData = {3, 2, -1, 3, -5, -12};

  EXPECT_EQ(result.size(), numCiphertextSlots);

  // check that provided values are in decryption result
  for (int i = 0; i < expectedData.size(); ++i) {
    EXPECT_EQ(expectedData.at(i), result.at(i));
  }
  // check all remaining ciphertext slots that contain result of (data1.last * data2.last)
  for (int i = expectedData.size(); i < result.size(); ++i) {
    ASSERT_EQ(data1.back() - data2.back(), result.at(i));
  }
}

#endif
