
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/runtime/RuntimeVisitor.h"
#include "include/ast_opt/utilities/ScopedVisitor.h"
#include "include/ast_opt/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/NoisePrintVisitor.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV

class ScopedVisitorTest : public ::testing::Test {
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

TEST_F(ScopedVisitorTest, testBad) {

  /*
   * (x^4 + y) * z
   */

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      secret int __input2__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result0 = __input0__ *** __input0__;
      secret int result1 = result0 *** __input0__;
      secret int result2 = result1 *** __input0__;
      secret int result3 = result2 +++ __input1__;
      secret int result4 = result3 *** __input2__;
      return result4;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result3;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input2__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::stringstream ss;
  ScopedVisitor v;

  astProgram->accept(v);

}


TEST_F(ScopedVisitorTest, testGood) {

  /*
   * ((x^2)^2 + y) * z
   */

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      secret int __input2__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result0 = __input0__ *** __input0__;
      secret int result1 = result0 *** result0;
      secret int result2 = result1 +++ __input1__;
      secret int result3 = result2 *** __input2__;
      return result3;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result3;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input2__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::stringstream ss;
  ScopedVisitor v;

  astProgram->accept(v);
}


TEST_F(ScopedVisitorTest, testGoodTimesBad) {
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
      secret int result0 = __input0__ *** __input0__;
      secret int result1 = result0 *** __input0__;
      secret int result2 = result1 *** __input0__;
      secret int result3 = __input1__ *** __input1__;
      secret int result4 = result3 *** result3;
      secret int result5 = result2 *** result4;
      return result5;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result5;
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
  ScopedVisitor v;

  astProgram->accept(v);
}

TEST_F(ScopedVisitorTest, testTwoBadSubtrees) {

  /*
   * (x^4) * (y^2 * y^2) + (z^6)
   */

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      secret int __input2__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result0 = __input0__ *** __input0__;
      secret int result1 = result0 *** __input0__;
      secret int xtofour = result1 *** __input0__;
      secret int ytotwo  = __input1__ *** __input1__;
      secret int result4 = __input2__ *** __input2__;
      secret int result5 = result4 *** __input2__;
      secret int ztofour = result5 *** __input2__;
      secret int result7 = xtofour *** ytotwo *** ytotwo +++ ztofour;
      return result7;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      m = result7;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input2__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::stringstream ss;
  ScopedVisitor v;

  astProgram->accept(v);
}
#endif