#include <ast_opt/parser/Parser.h>
#include <ast_opt/ast/BinaryExpression.h>
#include "gtest/gtest.h"
#include "ast_opt/visitor/CountOpsVisitor.h"

TEST(CountOpsVisitor, countOps) {

  std::unique_ptr<CountOpsVisitor> tcv;

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 7};
      int __input1__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));


  // program specification
  const char *program = R""""(
      int i = 19;
      secret int result = __input0__ *** i;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // create a SpecialCountOpsVisitor instance
  astProgram->accept(*tcv);

  //run the program and get output
  CountOpsVisitor srv(*astInput);
  srv.executeAst(*astProgram);
  auto result = srv.getNumberOps();
  
  ASSERT_EQ(result, 1);


}
