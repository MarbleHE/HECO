#include <ast_opt/visitor/runtime/SimulatorCiphertext.h>
#include "include/ast_opt/ast/AbstractNode.h"
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/ast/Literal.h"
#include "include/ast_opt/visitor/runtime/RuntimeVisitor.h"
#include "include/ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV


class RuntimeVisitorSimulatorTest : public ::testing::Test {
 protected:
  std::unique_ptr<SimulatorCiphertextFactory> scf;
  std::unique_ptr<TypeCheckingVisitor> tcv;

  void SetUp() override {
    scf = std::make_unique<SimulatorCiphertextFactory>(4096);
    tcv = std::make_unique<TypeCheckingVisitor>();
  }

  void registerInputVariable(Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
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

  // create ciphertexts to check noise heuristics
  std::vector<int64_t> data1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data1);
  double expected_noise = calcInitNoiseHeuristic(*ctxt);

  ASSERT_EQ(x.getNoise(), expected_noise);
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

  // create ciphertexts to check noise heuristics
  std::vector<int64_t> data1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);
  double expected_noise = calcAddNoiseHeuristic(*ctxt1, *ctxt2);

  ASSERT_EQ(x.getNoise(), expected_noise);
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

  // create ciphertexts to check noise heuristics
  std::vector<int64_t> data1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
  std::unique_ptr<AbstractCiphertext> ctxt1 = scf->createCiphertext(data1);
  std::vector<int64_t> data2 = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
  std::unique_ptr<AbstractCiphertext> ctxt2 = scf->createCiphertext(data2);
  double expected_noise = calcMultNoiseHeuristic(*ctxt1, *ctxt2);

  ASSERT_EQ(x.getNoise(), expected_noise);
}


#endif