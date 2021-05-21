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
  const int numCiphertextSlots = 8192;

  std::unique_ptr<SimulatorCiphertextFactory> scf;

  void SetUp() override {
    scf = std::make_unique<SimulatorCiphertextFactory>(numCiphertextSlots);
  }

  // calculates initial noise heuristic of a freshly encrypted ciphertext
  uint64_t calcInitNoiseHeuristic(AbstractCiphertext &abstractCiphertext) {
    uint64_t plain_modulus = scf->getContext().first_context_data()->parms().plain_modulus().value();
    uint64_t poly_modulus = scf->getContext().first_context_data()->parms().poly_modulus_degree();
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_t plain_mod;
    mpz_init(plain_mod);
    mpz_init_set_ui(plain_mod, plain_modulus);
    mpz_t poly_mod;
    mpz_init(poly_mod);
    mpz_init_set_ui(poly_mod, poly_modulus);
    // summand_one = n * (t-1) / 2
    mpz_t summand_one;
    mpz_init(summand_one);
    mpz_sub_ui(summand_one, plain_mod, 1);
    mpz_mul(summand_one, summand_one, poly_mod);
    mpz_div_ui(summand_one, summand_one, 2);
    // summand_two = 2 * sigma * sqrt(12 * n ^2 + 9 * n)
    mpz_t summand_two;
    mpz_init(summand_two);
    mpz_pow_ui(summand_two, poly_mod, 2);
    mpz_mul_ui(summand_two, summand_two, 12);
    mpz_t poly_mod_times_nine;
    mpz_init(poly_mod_times_nine);
    mpz_mul_ui(poly_mod_times_nine, poly_mod, 9);
    mpz_add(summand_two, summand_two, poly_mod_times_nine);
    mpz_sqrt(summand_two, summand_two);
    mpz_mul_ui(summand_two, summand_two, long(6.4)); // sigma = 3.2
    mpz_t sum;
    // sum = summand_1 + summand_2
    mpz_init(sum);
    mpz_add(sum, summand_one, summand_two);
    // result_noise = t * sum
    mpz_mul(result_noise, sum, plain_mod);
    size_t coeff_modulus_significant_bit_count = scf->getContext().first_context_data()->total_coeff_modulus_bit_count();
    size_t log_noise = mpz_sizeinbase(result_noise, 2);
    return std::max(int(coeff_modulus_significant_bit_count - log_noise - 1), 0);
  }

  // calculates noise heuristics for ctxt-ctxt add
  uint64_t calcAddNoiseHeuristic(AbstractCiphertext &abstractCiphertext1, AbstractCiphertext &abstractCiphertext2) {
    uint64_t result;
    auto &ctxt1 = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext1);
    auto &ctxt2 = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext2);
    // calc init noise heur for ctxt1 and ctxt2
    uint64_t plain_modulus = scf->getContext().first_context_data()->parms().plain_modulus().value();
    uint64_t poly_modulus = scf->getContext().first_context_data()->parms().poly_modulus_degree();
    mpz_t init_noise;
    mpz_init(init_noise);
    mpz_t plain_mod;
    mpz_init(plain_mod);
    mpz_init_set_ui(plain_mod, plain_modulus);
    mpz_t poly_mod;
    mpz_init(poly_mod);
    mpz_init_set_ui(poly_mod, poly_modulus);
    // summand_one = n * (t-1) / 2
    mpz_t summand_one;
    mpz_init(summand_one);
    mpz_sub_ui(summand_one, plain_mod, 1);
    mpz_mul(summand_one, summand_one, poly_mod);
    mpz_div_ui(summand_one, summand_one, 2);
    // summand_two = 2 * sigma * sqrt(12 * n ^2 + 9 * n)
    mpz_t summand_two;
    mpz_init(summand_two);
    mpz_pow_ui(summand_two, poly_mod, 2);
    mpz_mul_ui(summand_two, summand_two, 12);
    mpz_t poly_mod_times_nine;
    mpz_init(poly_mod_times_nine);
    mpz_mul_ui(poly_mod_times_nine, poly_mod, 9);
    mpz_add(summand_two, summand_two, poly_mod_times_nine);
    mpz_sqrt(summand_two, summand_two);
    mpz_mul_ui(summand_two, summand_two, long(6.4)); // sigma = 3.2
    mpz_t sum;
    // sum = summand_1 + summand_2
    mpz_init(sum);
    mpz_add(sum, summand_one, summand_two);
    // result_noise = t * sum
    mpz_mul(init_noise, sum, plain_mod);
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_add(result_noise, init_noise, init_noise);
    size_t coeff_modulus_significant_bit_count = scf->getContext().first_context_data()->total_coeff_modulus_bit_count();
    size_t log_noise = mpz_sizeinbase(result_noise, 2);
    return std::max(int(coeff_modulus_significant_bit_count - log_noise - 1), 0);
  }

  // calculates noise heuristics for ctxt-ctxt mult
  uint64_t calcMultNoiseHeuristic(AbstractCiphertext &abstractCiphertext1, AbstractCiphertext &abstractCiphertext2) {
    auto &ctxt1 = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext1);
    auto &ctxt2 = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext2);
    uint64_t plain_modulus = scf->getContext().first_context_data()->parms().plain_modulus().value();
    uint64_t poly_modulus = scf->getContext().first_context_data()->parms().poly_modulus_degree();
    uint64_t coeff_modulus = *scf->getContext().first_context_data()->total_coeff_modulus();
    auto ctx1_init_noise = calcInitNoiseHeuristic(ctxt1);
    auto ctx2_init_noise = calcInitNoiseHeuristic(ctxt2);
    mpz_t noise_1;
    mpz_init(noise_1);
    mpz_init_set_ui(noise_1, ctx1_init_noise);
    mpz_t noise_2;
    mpz_init(noise_2);
    mpz_init_set_ui(noise_2, ctx2_init_noise);
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_t plain_mod;
    mpz_init(plain_mod);
    mpz_init_set_ui(plain_mod, plain_modulus);
    mpz_t poly_mod;
    mpz_init(poly_mod);
    mpz_init_set_ui(poly_mod, poly_modulus);
    mpz_t coeff_mod;
    mpz_init(coeff_mod);
    mpz_init_set_ui(coeff_mod, coeff_modulus);
    // some powers of n and other things
    mpz_t poly_mod_squared;
    mpz_init(poly_mod_squared);
    mpz_pow_ui(poly_mod_squared, poly_mod, 2);
    mpz_t poly_mod_cubed;
    mpz_init(poly_mod_cubed);
    mpz_pow_ui(poly_mod_cubed, poly_mod, 3);
    mpz_t poly_mod_squared_times_two;
    mpz_init(poly_mod_squared_times_two);
    mpz_mul_ui(poly_mod_squared_times_two, poly_mod, 2);
    mpz_t poly_mod_cubed_times_four;
    mpz_init(poly_mod_cubed_times_four);
    mpz_mul_ui(poly_mod_cubed_times_four, poly_mod_cubed, 4);
    mpz_t poly_mod_cubed_times_four_div_three;
    mpz_init(poly_mod_cubed_times_four_div_three);
    mpz_div_ui(poly_mod_cubed_times_four_div_three, poly_mod_cubed_times_four, 3);
    // summand_one = t * sqrt(3n + 2n^2) (v1 + v2)
    mpz_t summand_one;
    mpz_init(summand_one);
    mpz_mul_ui(summand_one, poly_mod, 3);
    mpz_add(summand_one, summand_one, poly_mod_squared_times_two);
    mpz_sqrt(summand_one, summand_one);
    mpz_mul(summand_one, summand_one, plain_mod);
    mpz_t noise_sum;
    mpz_init(noise_sum);
    mpz_add(noise_sum, noise_1, noise_2);
    mpz_mul(summand_one, summand_one, noise_sum);
    //summand_two = 3 * v1 * v2 / q
    mpz_t summand_two;
    mpz_init(summand_two);
    mpz_mul(summand_two, noise_1, noise_2);
    mpz_mul_ui(summand_two, summand_two, 3);
    mpz_div(summand_two, summand_two, coeff_mod);
    //summand_three = t * sqrt(3d+2d^2+4d^3/3)
    mpz_t summand_three;
    mpz_init(summand_three);
    mpz_mul_ui(summand_three, poly_mod, 3);
    mpz_add(summand_three, summand_three, poly_mod_squared_times_two);
    mpz_add(summand_three, summand_three, poly_mod_cubed_times_four_div_three);
    mpz_sqrt(summand_three, summand_three);
    mpz_mul(summand_three, summand_three, plain_mod);
    // result_noise = summand_1 * summand_2 + summand_3
    mpz_add(result_noise, summand_one, summand_two);
    mpz_add(result_noise, result_noise, summand_three);
    size_t coeff_modulus_significant_bit_count = scf->getContext().first_context_data()->total_coeff_modulus_bit_count();
    size_t log_noise = mpz_sizeinbase(result_noise, 2);
    return std::max(int(coeff_modulus_significant_bit_count - log_noise - 1), 0);
  }

  uint64_t calcAddPlainNoiseHeuristic(AbstractCiphertext &abstractCiphertext, ICleartext &operand) {
    //noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
    uint64_t result;
    auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand);
    std::unique_ptr<seal::Plaintext> plaintext = scf->createPlaintext(cleartextInt->getData());
    auto &ctxt = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext);
//    double old_noise = ctxt.getNoise();
    int64_t rtq = scf->getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
    int64_t plain_max_abs_value = plaintext_norm(*plaintext);
    int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
//    result = old_noise + rtq * plain_max_coeff_count * plain_max_abs_value;
    return result;
  }

  uint64_t calcMultiplyPlainNoiseHeuristic(AbstractCiphertext &abstractCiphertext, ICleartext &operand) {
    uint64_t result;
    auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand);
    std::unique_ptr<seal::Plaintext> plaintext = scf->createPlaintext(cleartextInt->getData());
    auto &ctxt = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext);
//    uint64_t old_noise = ctxt.getNoise();
    int64_t plain_max_abs_value = plaintext_norm(*plaintext);
    int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  //  result = old_noise * plain_max_coeff_count * plain_max_abs_value;
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
    uint64_t result = (dynamic_cast<const SimulatorCiphertext&>(abstractCiphertext)).noiseBits();
    EXPECT_EQ(result, expected_noise);
  }

  int getCurrentNoiseBudget(const AbstractCiphertext &abstractCiphertext) {
    return (dynamic_cast<const SimulatorCiphertext&>(abstractCiphertext)).noiseBits();
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
  uint64_t expected_noise = calcInitNoiseHeuristic(*ctxt);
  std::cout << (dynamic_cast<const SimulatorCiphertext&>(*ctxt)).noiseBits();
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
  uint64_t expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);
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
  uint64_t expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);
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
  uint64_t expected_noise = calcMultNoiseHeuristic(*ctxt1, *ctxt2);
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

  uint64_t expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);

  ctxt1->addInplace(*ctxt2);
  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, subtractInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  uint64_t expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);
  ctxt1->subtractInplace(*ctxt2);
  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, multiplyInplace) { /* NOLINT */
  // create ciphertexts
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);

  uint64_t expected_noise = calcMultNoiseHeuristic(*ctxt1, *ctxt2);
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

  uint64_t expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);

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
  uint64_t expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);
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
 // uint64_t expected_noise = calcMultiplyPlainNoiseHeuristic(*ctxt1,operandVector); //TODO: check if this works
  //checkCiphertextNoise(*ctxtResult, expected_noise);

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

  uint64_t expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);

  ctxt1->addPlainInplace(operandVector);

  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, subPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextSim(data2);

  uint64_t expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);
  ctxt1->subtractPlainInplace(operandVector);
  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, multiplyPlainInplace) { /* NOLINT */
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  auto operandVector = createCleartextSim(data2);


  uint64_t expected_noise = calcMultiplyPlainNoiseHeuristic(*ctxt1,operandVector);
  ctxt1->multiplyPlainInplace(operandVector);
  checkCiphertextNoise(*ctxt1, expected_noise);
}

TEST_F(SimulatorCiphertextFactoryTest, testNoiseBIts) { /* NOLINT */
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  uint64_t tmp = round(log2(calcInitNoiseHeuristic(*ctxt1)));
  int expected_result = std::max(int((dynamic_cast<const SimulatorCiphertext&>(*ctxt1)).getFactory().getContext().first_context_data()->total_coeff_modulus_bit_count()
      - tmp - 1), 0);
  int result = (dynamic_cast<const SimulatorCiphertext&>(*ctxt1)).noiseBits();
  ASSERT_EQ(result, expected_result);
}


TEST_F(SimulatorCiphertextFactoryTest, xToPowerFourTimesYBad) {
  // create ciphertext
  std::vector<int64_t> data1 = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data1);

  std::vector<int> data2 = {0, 1, 2, 1, 10, 21};
  std::unique_ptr<AbstractCiphertext> ctxt3 = scf->createCiphertext(data2);

  //std::cout << "Noise(x): " << (dynamic_cast<const SimulatorCiphertext&>(*ctxt1)).noiseBits() << std::endl;
  // x * x
  ctxt1->multiplyInplace(*ctxt1);
  //std::cout << "Noise(x * x): " << (dynamic_cast<const SimulatorCiphertext&>(*ctxt1)).noiseBits() << std::endl;
  // x * x * x
  ctxt1->multiplyInplace(*ctxt1);
  //std::cout << "Noise(x * x * x): " << (dynamic_cast<const SimulatorCiphertext&>(*ctxt1)).noiseBits() << std::endl;
  //  x * x * x * x
  ctxt1->multiplyInplace(*ctxt1);
  //std::cout << "Noise(x * x * x): " << (dynamic_cast<const SimulatorCiphertext&>(*ctxt1)).noiseBits() << std::endl;
  ctxt1->multiplyInplace(*ctxt1);
  ctxt1->multiplyInplace(*ctxt1);
  ctxt1->multiplyInplace(*ctxt1);
  ctxt1->multiplyInplace(*ctxt1);
  ctxt1->multiplyInplace(*ctxt1);

}


#endif