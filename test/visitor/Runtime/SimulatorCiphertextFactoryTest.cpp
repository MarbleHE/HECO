#include <algorithm>

#include "gtest/gtest.h"

#include "include/ast_opt/visitor/runtime/SimulatorCiphertext.h"
#include "include/ast_opt/ast/ExpressionList.h"
#include "include/ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "include/ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/utilities/PlaintextNorm.h"

#ifdef HAVE_SEAL_BFV

class SimulatorCiphertextFactoryTest : public ::testing::Test {
 protected:
  const int numCiphertextSlots = 4096;

  std::unique_ptr<SimulatorCiphertextFactory> scf;

  void SetUp() override {
    scf = std::make_unique<SimulatorCiphertextFactory>(numCiphertextSlots);
  }

  // calculates initial noise heuristic of a freshly encrypted cipheertext
  double calcInitNoiseHeuristic(AbstractCiphertext &abstractCiphertext) {
    double result;
    auto &ctxt = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext);
    seal::Plaintext ptxt = ctxt.getPlaintext();
    int64_t plain_modulus = scf->getContext().first_context_data()->parms().plain_modulus().value();
    int64_t poly_modulus = scf->getContext().first_context_data()->parms().poly_modulus_degree();
    result = plain_modulus  * (poly_modulus * (plain_modulus - 1) / 2
        + 2 * 3.2 *sqrt(12 * pow(poly_modulus,2) + 9 * poly_modulus));
    return result;
  }

  // calculates noise heuristics for ctxt-ctxt add
  double calcAddNoiseHeuristic(AbstractCiphertext &abstractCiphertext1, AbstractCiphertext &abstractCiphertext2) {
    double result;
    auto &ctxt1 = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext1);
    auto &ctxt2 = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext2);
    result = ctxt1.getNoise() + ctxt2.getNoise();
    return result;
  }

  // calculates noise heuristics for ctxt-ctxt mult
  double calcMultNoiseHeuristic(AbstractCiphertext &abstractCiphertext1, AbstractCiphertext &abstractCiphertext2) {
    double result;
    auto &ctxt1 = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext1);
    auto &ctxt2 = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext2);
    int64_t plain_modulus = scf->getContext().first_context_data()->parms().plain_modulus().value();
    int64_t poly_modulus = scf->getContext().first_context_data()->parms().poly_modulus_degree();
    // iliashenko mult noise heuristic
    result =  plain_modulus * sqrt(3 * poly_modulus + 2 * pow(poly_modulus,2) )
        * (ctxt1.getNoise() + ctxt2.getNoise()) + 3 * ctxt1.getNoise() * ctxt2.getNoise() +
        plain_modulus * sqrt(3 * poly_modulus + 2 * pow(poly_modulus,2) +
            4 * pow(poly_modulus,3) /3);
    return result;
  }

  double calcAddPlainNoiseHeuristic(AbstractCiphertext &abstractCiphertext, ICleartext &operand) {
    //noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
    double result;
    auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand);
    std::unique_ptr<seal::Plaintext> plaintext = scf->createPlaintext(cleartextInt->getData());
    auto &ctxt = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext);
    double old_noise = ctxt.getNoise();
    int64_t rtq = scf->getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
    int64_t plain_max_abs_value = plaintext_norm(*plaintext);
    int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    result = old_noise + rtq * plain_max_coeff_count * plain_max_abs_value;
    return result;
  }

  double calcMultiplyPlainNoiseHeuristic(AbstractCiphertext &abstractCiphertext, ICleartext &operand) {
    double result;
    auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand);
    std::unique_ptr<seal::Plaintext> plaintext = scf->createPlaintext(cleartextInt->getData());
    auto &ctxt = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext);
    double old_noise = ctxt.getNoise();
    int64_t plain_max_abs_value = plaintext_norm(*plaintext);
    int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    result = old_noise * plain_max_coeff_count * plain_max_abs_value;
    return result;
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

  void checkCiphertextNoise(const AbstractCiphertext &abstractCiphertext, double expected_noise) {
    // get noise from input ciphertext and compare with the expected value
    double result = (dynamic_cast<const SimulatorCiphertext&>(abstractCiphertext)).getNoise();
    EXPECT_EQ(result, expected_noise);
    std::cout << (dynamic_cast<const SimulatorCiphertext&>(abstractCiphertext)).noiseBits();
  }
};

TEST_F(SimulatorCiphertextFactoryTest, createCiphertext) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);
  checkCiphertextData(*ctxt, data);
}

// =======================================
// == create fresh ciphertext
// =======================================

TEST_F(SimulatorCiphertextFactoryTest, createFresh) {
  // create ciphertexts
  std::vector<int64_t> data = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);

  double expected_noise = calcInitNoiseHeuristic(*ctxt);
  checkCiphertextNoise(*ctxt, expected_noise);
}

// =======================================
// == CTXT-CTXT operations with returned result
// =======================================

TEST_F(SimulatorCiphertextFactoryTest, add) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->add(*ctxt2);
  double expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);
  checkCiphertextNoise(*ctxtResult, expected_noise);

  // make sure that operands are not changed
  checkCiphertextData(*ctxt1, data1);
  checkCiphertextData(*ctxt2, data2);
}

TEST_F(SimulatorCiphertextFactoryTest, sub) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->subtract(*ctxt2);
  double expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);
  checkCiphertextNoise(*ctxtResult, expected_noise);

  // make sure that operands are not changed
  checkCiphertextData(*ctxt1, data1);
  checkCiphertextData(*ctxt2, data2);
}

TEST_F(SimulatorCiphertextFactoryTest, multiply) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  auto ctxtResult = ctxt1->multiply(*ctxt2);
  double expected_noise = calcMultNoiseHeuristic(*ctxt1, *ctxt2);
  checkCiphertextNoise(*ctxtResult, expected_noise);


  // make sure that operands are not changed
  checkCiphertextData(*ctxt1, data1);
  checkCiphertextData(*ctxt2, data2);
}

// =======================================
// == CTXT-CTXT in-place operations
// =======================================

TEST_F(SimulatorCiphertextFactoryTest, addInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  double expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);

  ctxt1->addInplace(*ctxt2);
  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, subtractInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  double expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);
  ctxt1->subtractInplace(*ctxt2);
  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, multiplyInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  double expected_noise = calcMultNoiseHeuristic(*ctxt1, *ctxt2);
  ctxt1->multiplyInplace(*ctxt2);
  checkCiphertextNoise(*ctxt1, expected_noise);
}

// =======================================
// == CTXT-PLAIN operations with returned result
// =======================================

Cleartext<int> createCleartextSim(const std::vector<int> &literalIntValues) {
  std::vector<std::unique_ptr<AbstractExpression>> result;
  for (const auto &val : literalIntValues) {
    result.emplace_back(std::make_unique<LiteralInt>(val));
  }
  return Cleartext<int>(literalIntValues);
}

TEST_F(SimulatorCiphertextFactoryTest, addPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextSim(data2);

  double expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);

  auto ctxtResult = ctxt1->addPlain(operandVector);

  checkCiphertextNoise(*ctxtResult, expected_noise);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

TEST_F(SimulatorCiphertextFactoryTest, subPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextSim(data2);

  auto ctxtResult = ctxt1->subtractPlain(operandVector);
  double expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);
  checkCiphertextNoise(*ctxtResult, expected_noise);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

TEST_F(SimulatorCiphertextFactoryTest, multiplyPlain) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextSim(data2);

  auto ctxtResult = ctxt1->multiplyPlain(operandVector);
  double expected_noise = calcMultiplyPlainNoiseHeuristic(*ctxt1,operandVector); //TODO: check if this works
  checkCiphertextNoise(*ctxtResult, expected_noise);

  // make sure that ciphertext operand is not changed
  checkCiphertextData(*ctxt1, data1);
}

// =======================================
// == CTXT-PLAIN in-place operations
// =======================================

TEST_F(SimulatorCiphertextFactoryTest, addPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextSim(data2);

  double expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);

  ctxt1->addPlainInplace(operandVector);

  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, subPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextSim(data2);

  double expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);
  ctxt1->subtractPlainInplace(operandVector);
  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, multiplyPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextSim(data2);


  double expected_noise = calcMultiplyPlainNoiseHeuristic(*ctxt1,operandVector);
  ctxt1->multiplyPlainInplace(operandVector);
  checkCiphertextNoise(*ctxt1, expected_noise);
}



#endif