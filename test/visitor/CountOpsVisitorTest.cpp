#include <ast_opt/parser/Parser.h>
#include <ast_opt/ast/BinaryExpression.h>
#include "gtest/gtest.h"
#include "ast_opt/visitor/CountOpsVisitor.h"


class CountOpsVisitorTest: public ::testing::Test {
 protected:

  void SetUp() override {
  }

};

TEST_F(CountOpsVisitorTest, countOps) {

  // program specification
  const char *program = R""""(
      int __input0__ = 0;
      int i = 19;
      int result = __input0__ *** i;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));


  // create a CountOpsVisitor instance
  auto countOpsVisitor = CountOpsVisitor();
  auto rootScope = std::make_unique<Scope>(*astProgram);
  countOpsVisitor.setRootScope(std::move(rootScope));

  // start visiting
  astProgram->accept(countOpsVisitor);

  ASSERT_EQ(countOpsVisitor.getNumberOps(), 1);
}
