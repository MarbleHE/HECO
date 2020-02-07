#include <Ast.h>
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"
#include "Function.h"
#include "BinaryExpr.h"

class AstTestFixture : public ::testing::Test {
 protected:
  Ast ast;

  void SetUp() override {
    AstTestingGenerator::generateAst(7, ast);
  }
};

TEST_F(AstTestFixture, deleteNodeFromAst_deleteSingleLeafNodeOnly) {
  // retrieve the binary expression we are interested in
  auto func = dynamic_cast<Function*>(ast.getRootNode());
  ASSERT_NE(func, nullptr);
  auto binaryExpr = dynamic_cast<BinaryExpr*>(func->getBody().at(0)->getChildAtIndex(1)->getChildAtIndex(2));
  ASSERT_NE(binaryExpr, nullptr);

  // retrieve the deletion target -> variable of the binary expression
  auto variable = binaryExpr->getChildAtIndex(0);
  ASSERT_NE(variable, nullptr);
  ASSERT_EQ(variable->getParents().front(), binaryExpr);

  variable = ast.deleteNode(variable);
  ASSERT_EQ(binaryExpr->getChildAtIndex(0), nullptr);
  ASSERT_EQ(variable, nullptr);
}

TEST(AstClassTest, deleteNodeFromAst_deleteRecursiveSubtreeFull) {

}

TEST(AstClassTest, deleteNodeFromAst_deleteRecursiveSubtreeEmpty) {

}

TEST(AstClassTest, deleteNodeFromAst_ChildrenExisting) {

}

