#include <gtest/gtest.h>
#include <include/ast_opt/ast/Ast.h>
#include <include/ast_opt/visitor/PrintVisitor.h>
#include <include/ast_opt/visitor/CompileTimeExpressionSimplifier.h>
#include <include/ast_opt/optimizer/BatchingChecker.h>
#include "AstTestingGenerator.h"

TEST(BatchingChecker, testAst) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(61, ast);

  CompileTimeExpressionSimplifier ctes;
  ctes.visit(ast);
//
//  PrintVisitor pv;
//  pv.useUniqueNodeIds(true);
//  pv.visit(ast);

  BatchingChecker::determineBatchability(ast.getRootNode());
}
