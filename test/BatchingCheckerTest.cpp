#include <gtest/gtest.h>
#include <include/ast_opt/ast/Ast.h>
#include <include/ast_opt/visitor/PrintVisitor.h>
#include <include/ast_opt/visitor/CompileTimeExpressionSimplifier.h>
#include <include/ast_opt/optimizer/BatchingChecker.h>
#include <include/ast_opt/ast/Function.h>
#include <include/ast_opt/ast/MatrixAssignm.h>
#include <include/ast_opt/ast/VarAssignm.h>
#include <include/ast_opt/ast/Return.h>
#include "AstTestingGenerator.h"

TEST(BatchingChecker, laplacianAstInnerLoopsWithNonStdWeights) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(61, ast);

  CompileTimeExpressionSimplifier ctes;
  ctes.visit(ast);

  // find all assignments targets (rvalue-like) in MatrixAssignm and VarAssignm statements
  auto func = ast.getRootNode()->castTo<Function>();
  std::vector<AbstractExpr *> statementsTopExprs;
  for (auto node : ast.getAllNodes()) {
    if (auto ma = dynamic_cast<MatrixAssignm *>(node)) {
      statementsTopExprs.push_back(ma->getValue());
    } else if (auto va = dynamic_cast<VarAssignm *>(node)) {
      statementsTopExprs.push_back(va->getValue());
    }
  }

  // determine the largest batchable subtree using the found expressions
  AbstractNode *rootOfLargestBatchableSubtree;
  for (auto expr : statementsTopExprs) {
    rootOfLargestBatchableSubtree = BatchingChecker::getLargestBatchableSubtree(expr);
    if (rootOfLargestBatchableSubtree!=nullptr) break;
  }

  EXPECT_EQ(rootOfLargestBatchableSubtree->getNodeType(), OperatorExpr().getNodeType());
  EXPECT_EQ(rootOfLargestBatchableSubtree->countChildrenNonNull(), 10);
  EXPECT_EQ(rootOfLargestBatchableSubtree->getDescendants().size(), 142);
  EXPECT_TRUE(rootOfLargestBatchableSubtree->castTo<OperatorExpr>()->getOperator()->equals(ADDITION));
  EXPECT_TRUE(BatchingChecker::shouldBeBatched(rootOfLargestBatchableSubtree));
}

TEST(BatchingChecker, laplacianAstFullLoopsWithStdWeights) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(60, ast);

  CompileTimeExpressionSimplifier ctes((CtesConfiguration(2)));
  ctes.visit(ast);

  // find all assignments targets (rvalue-like) in MatrixAssignm and VarAssignm statements
  auto func = ast.getRootNode()->castTo<Function>();
  std::vector<AbstractExpr *> statementsTopExprs;
  for (auto node : ast.getAllNodes()) {
    if (auto ma = dynamic_cast<MatrixAssignm *>(node)) {
      statementsTopExprs.push_back(ma->getValue());
    } else if (auto va = dynamic_cast<VarAssignm *>(node)) {
      statementsTopExprs.push_back(va->getValue());
    }
  }

  // determine the largest batchable subtree using the found expressions
  AbstractNode *rootOfLargestBatchableSubtree;
  for (auto expr : statementsTopExprs) {
    rootOfLargestBatchableSubtree = BatchingChecker::getLargestBatchableSubtree(expr);
    if (rootOfLargestBatchableSubtree!=nullptr) break;
  }

  EXPECT_EQ(rootOfLargestBatchableSubtree->getNodeType(), OperatorExpr().getNodeType());
  EXPECT_EQ(rootOfLargestBatchableSubtree->countChildrenNonNull(), 3);
  EXPECT_EQ(rootOfLargestBatchableSubtree->getDescendants().size(), 12);
  EXPECT_TRUE(rootOfLargestBatchableSubtree->castTo<OperatorExpr>()->getOperator()->equals(MULTIPLICATION));
  EXPECT_FALSE(BatchingChecker::shouldBeBatched(rootOfLargestBatchableSubtree));
}
