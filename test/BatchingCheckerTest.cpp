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

TEST(BatchingChecker, testAst) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(61, ast);

  CompileTimeExpressionSimplifier ctes;
  ctes.visit(ast);

  // uncomment to see how the CTES'ed AST looks like
//  PrintVisitor pv;
//  pv.useUniqueNodeIds(true);
//  pv.visit(ast);

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

  // determine the largest batchable subtree for each found expression
  std::cout << "##############################################" << std::endl;
  for (auto expr : statementsTopExprs) {
    auto node = BatchingChecker::getLargestBatchableSubtree(expr);
    if (node!=nullptr) {
      std::cout << "-- subtree rooted in " << node->getUniqueNodeId() << std::endl;
    }
  }
  std::cout << "##############################################" << std::endl;;
}
