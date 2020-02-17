#include "Ast.h"
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"
#include "Function.h"
#include "BinaryExpr.h"

class AstTestFixture : public ::testing::Test {
 protected:
  Ast ast;

  void SetUp() override {
    // This AST corresponds to the program:
    // int computePrivate(int inputA, int inputB, int inputC) {
    //   int prod = inputA * [inputB * inputC]
    //   return prod / 3;
    // }
    AstTestingGenerator::generateAst(7, ast);
  }
};

TEST_F(AstTestFixture, deleteNode_deleteSingleLeafNodeOnly) { /* NOLINT */
  // retrieve the binary expression of interest
  auto func = dynamic_cast<Function *>(ast.getRootNode());
  ASSERT_NE(func, nullptr);
  auto binaryExpr = dynamic_cast<BinaryExpr *>(func->getBody().at(0)->getChildAtIndex(1)->getChildAtIndex(2));
  ASSERT_NE(binaryExpr, nullptr);

  // retrieve the deletion target -> variable of the binary expression
  auto variable = binaryExpr->getChildAtIndex(0);
  ASSERT_NE(variable, nullptr);
  ASSERT_EQ(variable->getParents().front(), binaryExpr);

  // delete node and verify deletion success
  ast.deleteNode(&variable);
  ASSERT_EQ(binaryExpr->getChildAtIndex(0), nullptr);
  ASSERT_EQ(variable, nullptr);
}

TEST_F(AstTestFixture, deleteNode_deleteRecursiveSubtreeNonEmpty) { /* NOLINT */
  // retrieve the binary expression of interest
  auto func = dynamic_cast<Function *>(ast.getRootNode());
  ASSERT_NE(func, nullptr);
  auto binaryExprParent = func->getBody().at(0)->getChildAtIndex(1);
  auto binaryExpr = dynamic_cast<BinaryExpr *>(binaryExprParent->getChildAtIndex(2));
  ASSERT_NE(binaryExpr, nullptr);

  // save and check node's children
  const auto &binaryExprChildren = binaryExpr->getChildren();
  for (auto &c : binaryExprChildren)
    ASSERT_EQ(c->getParentsNonNull().front(), binaryExpr);

  // delete node and its subtree and verify deletion success
  Node *binaryExprPtr = binaryExpr;
  ast.deleteNode(&binaryExprPtr, true);
  // verify that BinaryExpr was deleted, also from its parent
  ASSERT_EQ(binaryExprPtr, nullptr);
  ASSERT_EQ(binaryExprParent->getChildAtIndex(2), nullptr);
  // verify that children are deleted
  ASSERT_TRUE(binaryExprChildren.empty());
}

TEST_F(AstTestFixture, deleteNode_deleteRecursiveSubtreeEmpty) { /* NOLINT */
  // The same test as deleteNode_deleteSingleLeafNodeOnly but now we use the 'deleteSubtreeRecursively' flag
  // this should not change anything though if there are no children present

  // retrieve the binary expression of interest
  auto func = dynamic_cast<Function *>(ast.getRootNode());
  ASSERT_NE(func, nullptr);
  auto binaryExpr = dynamic_cast<BinaryExpr *>(func->getBody().at(0)->getChildAtIndex(1)->getChildAtIndex(2));
  ASSERT_NE(binaryExpr, nullptr);

  // retrieve the deletion target -> variable of the binary expression
  auto variable = binaryExpr->getChildAtIndex(0);
  ASSERT_NE(variable, nullptr);
  ASSERT_EQ(variable->getParents().front(), binaryExpr);

  // delete node and verify deletion success
  ast.deleteNode(&variable, true);
  ASSERT_EQ(binaryExpr->getChildAtIndex(0), nullptr);
  ASSERT_EQ(variable, nullptr);
}

TEST_F(AstTestFixture, deleteNode_ChildrenExisting) { /* NOLINT */
  // The same test as deleteNode_deleteRecursiveSubtreeNonEmpty but now we simulate forgetting the use of the
  // 'deleteSubtreeRecursively' flag which should throw an exception

  // retrieve the binary expression of interest
  auto func = dynamic_cast<Function *>(ast.getRootNode());
  ASSERT_NE(func, nullptr);
  auto binaryExprParent = func->getBody().at(0)->getChildAtIndex(1);
  auto binaryExpr = dynamic_cast<BinaryExpr *>(binaryExprParent->getChildAtIndex(2));
  ASSERT_NE(binaryExpr, nullptr);

  // save and check node's children
  int binaryExprNumChildren = binaryExpr->countChildrenNonNull();
  for (auto &c : binaryExpr->getChildrenNonNull())
    ASSERT_EQ(c->getParentsNonNull().front(), binaryExpr);

  // delete node and its subtree and verify deletion success
  Node *binaryExprPtr = binaryExpr;
  // by using the default parameter value for deleteSubtreeRecursively
  EXPECT_THROW(ast.deleteNode(&binaryExprPtr), std::logic_error);
  // by expliciting passing the parameter value for deleteSubtreeRecursively
  EXPECT_THROW(ast.deleteNode(&binaryExprPtr, false), std::logic_error);

  // verify that BinaryExpr was not deleted, also from its parent
  ASSERT_NE(binaryExprPtr, nullptr);
  ASSERT_EQ(binaryExprParent->getChildAtIndex(2), binaryExpr);
  // verify that children are deleted
  ASSERT_EQ(binaryExpr->countChildrenNonNull(), binaryExprNumChildren);
}

TEST_F(AstTestFixture, deepCopy) { /* NOLINT */
  // Test that copying an AST object properly performs a deep copy
  auto number_of_nodes = ast.getAllNodes().size();
  if (number_of_nodes!=0) {
    Ast copy = ast;

    // Delete all nodes in the copy
    auto nodes = copy.getAllNodes();
    for (auto x : nodes) {
      copy.deleteNode(&x);
      ASSERT_EQ(x, nullptr);
    }

    // Ensure that the copy is empty
    ASSERT_TRUE(copy.getAllNodes().empty());

    // Ensure that original still has all nodes
    EXPECT_EQ(ast.getAllNodes().size(), number_of_nodes);

  } else {
    GTEST_SKIP_("Cannot perform deep copy test on empty AST");
  }
}
