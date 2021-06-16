#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/visitor/runtime/RuntimeVisitor.h"
#include "include/ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV

class IdentifyNoisySubtreeVisitorTest: public ::testing::Test {
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

};


TEST_F(IdentifyNoisySubtreeVisitorTest, testGoodTimesBad) {
  /*
   * (x*x*x*x) * (x^2 * x^2)
   */

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = (__input0__ ***  __input0__ *** __input0__ ***  __input0__) *** ((__input0__ ***  __input0__) *** (__input0__ ***  __input0__));
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

  std::stringstream ss;
  IdentifyNoisySubtreeVisitor v(ss, srv.getNoiseMap(), srv.getRelNoiseMap());

  astProgram->accept(v);

  std::cout << "Program: " << std::endl;
  std::cout << ss.str() << std::endl;

}
#endif