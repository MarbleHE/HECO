#include <ast_opt/parser/Parser.h>
#include <ast_opt/ast/BinaryExpression.h>
#include "gtest/gtest.h"
#include "include/ast_opt/visitor/runtime/RuntimeVisitor.h"
#include "ast_opt/visitor/CountOpsVisitor.h"


class RunTimeOpCounterTest: public ::testing::Test {
 protected:
  std::unique_ptr<TypeCheckingVisitor> tcv;

  void SetUp() override {
    tcv = std::make_unique<TypeCheckingVisitor>();
  }

  void registerInputVariable(Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  }
};

TEST_F(RunTimeOpCounterTest, countOps) {

  std::unique_ptr<TypeCheckingVisitor> tcv;
  tcv = std::make_unique<TypeCheckingVisitor>();

  // program's input
  const char *inputs = R""""(
      int __input0__ = 0;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      int __input0__ = 0;
      int i = 19;
      int result = __input0__ *** i;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));


  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));

  // create a SpecialCountOpsVisitor instance
  astProgram->accept(*tcv);

  //run the program and get output
  CountOpsVisitor srv(*astInput);
 /* srv.executeAst(*astProgram);
  auto result = srv.getNumberOps();

  ASSERT_EQ(result, 1);*/
}
