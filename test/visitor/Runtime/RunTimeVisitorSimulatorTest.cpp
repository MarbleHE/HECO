#include <ast_opt/visitor/runtime/SimulatorCiphertext.h>
#include "include/ast_opt/ast/AbstractNode.h"
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/ast/Literal.h"
#include "include/ast_opt/visitor/runtime/RuntimeVisitor.h"
#include "include/ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/utilities/PlaintextNorm.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV


class RuntimeVisitorSimulatorTest : public ::testing::Test {
 protected:
  std::unique_ptr<SimulatorCiphertextFactory> scf;
  std::unique_ptr<TypeCheckingVisitor> tcv;

  void SetUp() override {
    scf = std::make_unique<SimulatorCiphertextFactory>(8192);
    tcv = std::make_unique<TypeCheckingVisitor>();
  }

  void registerInputVariable(Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  }

  // calculates initial noise heuristic of a freshly encrypted ciphertext
  uint64_t calcInitNoiseHeuristic() {
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
  uint64_t calcAddNoiseHeuristic() {
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
  uint64_t calcMultNoiseHeuristic() {
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
    mpz_t coeff_mod;
    mpz_init(coeff_mod);
    mpz_init_set_ui(coeff_mod, *scf->getContext().first_context_data()->total_coeff_modulus());
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
    mpz_t sum_one;
    mpz_init(sum_one);
    mpz_mul_ui(sum_one, poly_mod, 3);
    mpz_add(sum_one, sum_one, poly_mod_squared_times_two);
    mpz_sqrt(sum_one, sum_one);
    mpz_mul(sum_one, sum_one, plain_mod);
    mpz_t noise_sum;
    mpz_init(noise_sum);
    mpz_add(noise_sum, init_noise, init_noise);
    mpz_mul(sum_one, sum_one, noise_sum);
    //summand_two = 3 * v1 * v2 / q
    mpz_t sum_two;
    mpz_init(sum_two);
    mpz_mul(sum_two, init_noise, init_noise);
    mpz_mul_ui(sum_two, sum_two, 3);
    mpz_div(sum_two, sum_two, coeff_mod);
    //summand_three = t * sqrt(3d+2d^2+4d^3/3)
    mpz_t summand_three;
    mpz_init(summand_three);
    mpz_mul_ui(summand_three, poly_mod, 3);
    mpz_add(summand_three, summand_three, poly_mod_squared_times_two);
    mpz_add(summand_three, summand_three, poly_mod_cubed_times_four_div_three);
    mpz_sqrt(summand_three, summand_three);
    mpz_mul(summand_three, summand_three, plain_mod);
    // result_noise = summand_1 * summand_2 + summand_3
    mpz_add(result_noise, sum_one, sum_two);
    mpz_add(result_noise, result_noise, summand_three);
    size_t coeff_modulus_significant_bit_count = scf->getContext().first_context_data()->total_coeff_modulus_bit_count();
    size_t log_noise = mpz_sizeinbase(result_noise, 2);
    return std::max(int(coeff_modulus_significant_bit_count - log_noise - 1), 0);
  }

  uint64_t calcAddPlainNoiseHeuristic(AbstractCiphertext &abstractCiphertext, ICleartext &operand) {
    //noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
    std::unique_ptr<seal::Plaintext> plaintext = scf->createPlaintext(dynamic_cast<Cleartext<int> *>(&operand)->getData());
    int64_t rtq = scf->getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
    int64_t plain_max_abs_value = plaintext_norm(*plaintext);
    int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_modulus = scf->getContext().first_context_data()->parms().plain_modulus().value();
    uint64_t poly_modulus = scf->getContext().first_context_data()->parms().poly_modulus_degree();
    // calc init noise
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
    // calc heuristic
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_t  rt_q;
    mpz_init(rt_q);
    mpz_init_set_ui(rt_q, rtq);
    mpz_t  plain_coeff_ct;
    mpz_init(plain_coeff_ct);
    mpz_init_set_ui(plain_coeff_ct, plain_max_coeff_count);
    mpz_t  plain_abs;
    mpz_init(plain_abs);
    mpz_init_set_ui(plain_abs, plain_max_abs_value);
    mpz_mul(result_noise, rt_q, plain_coeff_ct);
    mpz_mul(result_noise, result_noise, plain_abs);
    mpz_add(result_noise, result_noise, init_noise);
    size_t coeff_modulus_significant_bit_count = scf->getContext().first_context_data()->total_coeff_modulus_bit_count();
    size_t log_noise = mpz_sizeinbase(result_noise, 2);
    return std::max(int(coeff_modulus_significant_bit_count - log_noise - 1), 0);
  }

  uint64_t calcMultiplyPlainNoiseHeuristic(AbstractCiphertext &abstractCiphertext, ICleartext &operand) {
    std::unique_ptr<seal::Plaintext> plaintext = scf->createPlaintext(dynamic_cast<Cleartext<int> *>(&operand)->getData());
    int64_t rtq = scf->getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
    int64_t plain_max_abs_value = plaintext_norm(*plaintext);
    int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_modulus = scf->getContext().first_context_data()->parms().plain_modulus().value();
    uint64_t poly_modulus = scf->getContext().first_context_data()->parms().poly_modulus_degree();
    // calc init noise
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
    //calc heuristic
    // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
    mpz_t plain_coeff_ct;
    mpz_init(plain_coeff_ct);
    mpz_init_set_ui(plain_coeff_ct, plain_max_coeff_count);
    mpz_t plain_abs;
    mpz_init(plain_abs);
    mpz_init_set_ui(plain_abs, plain_max_abs_value);
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_mul(result_noise, plain_abs, plain_coeff_ct);
    mpz_mul(result_noise, result_noise, init_noise);
    size_t coeff_modulus_significant_bit_count = scf->getContext().first_context_data()->total_coeff_modulus_bit_count();
    size_t log_noise = mpz_sizeinbase(result_noise, 2);
    return std::max(int(coeff_modulus_significant_bit_count - log_noise - 1), 0);
  }

  void checkCiphertextNoise(const AbstractCiphertext &abstractCiphertext, double expected_noise) {
    std::cout << (dynamic_cast<const SimulatorCiphertext&>(abstractCiphertext)).noiseBits();
    EXPECT_EQ((dynamic_cast<const SimulatorCiphertext&>(abstractCiphertext)).noiseBits(), expected_noise);
  }
};

// =======================================
// == create fresh ciphertext
// =======================================

TEST_F(RuntimeVisitorSimulatorTest, testFreshCtxt) {
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = __input0__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
  auto result = srv.getOutput(*astOutput);

  auto x = dynamic_cast<SimulatorCiphertext &>(*result[0].second);

  uint64_t expected_noise = calcInitNoiseHeuristic();

  ASSERT_EQ( x.noiseBits(), expected_noise);
}

// =======================================
// == CTXT-CTXT operations with returned result
// =======================================

TEST_F(RuntimeVisitorSimulatorTest, testAddCtxtCtxt) {
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = __input0__ +++ __input1__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {1032, 34, 222, 4, 22, 44, 3825, 0, 1, 21};
  auto result = srv.getOutput(*astOutput);
  auto x = dynamic_cast<SimulatorCiphertext &>(*result[0].second);

  uint64_t expected_noise = calcAddNoiseHeuristic();

  ASSERT_EQ(x.noiseBits(), expected_noise);
}

TEST_F(RuntimeVisitorSimulatorTest, testSubCtxtCtxt) {
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = __input0__ --- __input1__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {1032, 34, 222, 4, 22, 44, 3825, 0, 1, 21};
  auto result = srv.getOutput(*astOutput);

  auto x = dynamic_cast<SimulatorCiphertext &>(*result[0].second);

  uint64_t expected_noise = calcAddNoiseHeuristic();

  ASSERT_EQ(x.noiseBits(), expected_noise);
}

TEST_F(RuntimeVisitorSimulatorTest, testMultCtxtCtxt) {
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = __input0__ *** __input1__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {1032, 34, 222, 4, 22, 44, 3825, 0, 1, 21};
  auto result = srv.getOutput(*astOutput);

  auto x = dynamic_cast<SimulatorCiphertext &>(*result[0].second);

  uint64_t expected_noise = calcMultNoiseHeuristic();
  ASSERT_EQ(x.noiseBits(), expected_noise);
}

// =======================================
// == CTXT-PLAIN operations with returned result
// =======================================

Cleartext<int> createCleartextSimVisitor(const std::vector<int> &literalIntValues) {
  std::vector<std::unique_ptr<AbstractExpression>> result;
  for (const auto &val : literalIntValues) {
    result.emplace_back(std::make_unique<LiteralInt>(val));
  }
  return Cleartext<int>(literalIntValues);
}

TEST_F(RuntimeVisitorSimulatorTest, testAddCtxtPlaintext) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,  22, 11, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int i = 19;
      secret int result = __input0__ +++  i;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
      x = result[3];
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {817, 19, 19, 418, 209, 133};
  expectedResult["x"] = {418};
  auto result = srv.getOutput(*astOutput);

  auto x = dynamic_cast<SimulatorCiphertext &>(*result[0].second);

  // create ciphertext and plaintext to check noise heuristics
  std::vector<int64_t> data1 = {43,  1,   1,  22, 11, 7};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  auto operandVector = createCleartextSimVisitor({19});
  uint64_t expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);
  ASSERT_EQ(x.noiseBits(), expected_noise);
}

TEST_F(RuntimeVisitorSimulatorTest, testSubCtxtPlaintext) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,  22, 11, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int i = 19;
      secret int result = __input0__ ---  i;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
      x = result[3];
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {817, 19, 19, 418, 209, 133};
  expectedResult["x"] = {418};
  auto result = srv.getOutput(*astOutput);

  auto x = dynamic_cast<SimulatorCiphertext &>(*result[0].second);

  // create ciphertext and plaintext to check noise heuristics
  std::vector<int64_t> data1 = {43,  1,   1,  22, 11, 7};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  auto operandVector = createCleartextSimVisitor({19});
  uint64_t expected_noise = calcAddPlainNoiseHeuristic(*ctxt1,operandVector);
  ASSERT_EQ(x.noiseBits(), expected_noise);
}

TEST_F(RuntimeVisitorSimulatorTest, testMultCtxtPlaintext) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,  22, 11, 7};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int i = 19;
      secret int result = __input0__ ***  i;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
      x = result[3];
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);
  auto secretTaintedNodesMap = tcv->getSecretTaintedNodes();

  // run the program and get its output
  RuntimeVisitor srv(*scf, *astInput, secretTaintedNodesMap);
  srv.executeAst(*astProgram);

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["y"] = {817, 19, 19, 418, 209, 133};
  expectedResult["x"] = {418};
  auto result = srv.getOutput(*astOutput);

  auto x = dynamic_cast<SimulatorCiphertext &>(*result[0].second);

  // create ciphertext and plaintext to check noise heuristics
  std::vector<int64_t> data1 = {43,  1,   1,  22, 11, 7};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  auto operandVector = createCleartextSimVisitor({19});
  uint64_t expected_noise = calcMultiplyPlainNoiseHeuristic(*ctxt1,operandVector);
  ASSERT_EQ(x.noiseBits(), expected_noise);
}
#endif