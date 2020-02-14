#include <gtest/gtest.h>
#include <Ast.h>
#include <include/visitor/MultDepthVisitor.h>
#include "Operator.h"
#include "Function.h"
#include "BinaryExpr.h"
#include "LogicalExpr.h"
#include "VarAssignm.h"
#include "AstTestingGenerator.h"

TEST(MultDepthVisitorTests, SingleStatementMultiplication) { // NOLINT
  // construct AST
  Ast ast;
  // void f() { int abc = 22 * 32; }
  auto* f = new Function("f");
  f->addStatement(new VarDecl("abc", TYPES::INT,
                              new BinaryExpr(
                                  new LiteralInt(22),
                                  OpSymb::multiplication,
                                  new LiteralInt(32))));
  ast.setRootNode(f);

  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);

  EXPECT_EQ(mdv.getMaxDepth(), 1);
}

TEST(MultDepthVisitorTests, NestedMultiplication) { // NOLINT
  // construct AST
  Ast ast;
  // void f() { int abc = 22 * (32 * 53); }
  auto* f = new Function("f");
  f->addStatement(new VarDecl("abc",
                              TYPES::INT,
                              new BinaryExpr(
                                  new LiteralInt(22),
                                  OpSymb::multiplication,
                                  new BinaryExpr(
                                      new LiteralInt(32),
                                      OpSymb::multiplication,
                                      new LiteralInt(53)))));
  ast.setRootNode(f);

  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);

  EXPECT_EQ(mdv.getMaxDepth(), 2);
}

TEST(MultDepthVisitorTests, MultipleStatementMultiplication) { // NOLINT
  // construct AST
  Ast ast;

  // void f(int num) { int alpha = 32 * num; int beta = alpha * 123; }
  auto* f = new Function("f");
  ast.setRootNode(f);
  f->addParameter(new FunctionParameter("int", new Variable("num")));

  f->addStatement(new VarDecl("alpha", TYPES::INT, new BinaryExpr(
      new LiteralInt(32),
      OpSymb::multiplication,
      new Variable("num"))));

  f->addStatement(new VarDecl("beta",
                              TYPES::INT,
                              new BinaryExpr(
                                  new Variable("alpha"),
                                  OpSymb::multiplication,
                                  new LiteralInt(123))));

  // calculate multiplicative depth
  MultDepthVisitor mdv(false);
  mdv.visit(ast);

  EXPECT_EQ(mdv.getMaxDepth(), 2);
}

TEST(MultDepthVisitorTests, SingleStatementLogicalAnd) { // NOLINT
  // construct AST
  Ast ast;
  // void f() { int abc = 22 * 32; }
  auto* f = new Function("f");
  f->addStatement(new VarDecl("abc", TYPES::INT,
                              new LogicalExpr(
                                  new LiteralBool(true),
                                  OpSymb::logicalAnd,
                                  new LiteralBool(false))));
  ast.setRootNode(f);

  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);

  EXPECT_EQ(mdv.getMaxDepth(), 1);
}

TEST(MultDepthVisitorTests, NestedStatementLogicalAnd) { // NOLINT
  // construct AST
  Ast ast;
  // void f() { int abc = 22 * (32 * 53); }
  auto* f = new Function("f");
  f->addStatement(new VarDecl("abc",
                              TYPES::BOOL,
                              new LogicalExpr(
                                  new LiteralBool(false),
                                  OpSymb::logicalAnd,
                                  new LogicalExpr(
                                      new LiteralBool(true),
                                      OpSymb::logicalAnd,
                                      new LiteralBool(false)))));
  ast.setRootNode(f);

  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);

  EXPECT_EQ(mdv.getMaxDepth(), 2);
}

TEST(MultDepthVisitorTests, MultipleStatementsLogicalAnd) { // NOLINT
  // construct AST
  Ast ast;

  // void f(int num) { int alpha = 32 * num; int beta = alpha * 123; }
  auto* f = new Function("f");
  ast.setRootNode(f);
  f->addParameter(new FunctionParameter("int", new Variable("num")));

  f->addStatement(new VarDecl("alpha", TYPES::BOOL,
                              new LogicalExpr(
                                  new LiteralBool(true),
                                  OpSymb::logicalAnd,
                                  new LiteralBool(false))));

  f->addStatement(new VarDecl("beta", TYPES::BOOL,
                              new LogicalExpr(
                                  new Variable("alpha"),
                                  OpSymb::logicalAnd,
                                  new LiteralBool(true))));

  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);

  EXPECT_EQ(mdv.getMaxDepth(), 2);
}

TEST(MultDepthVisitorTests, NoLogicalAndOrMultiplicationPresent) { // NOLINT
  Ast ast;
  auto* f = new Function("f");
  ast.setRootNode(f);
  f->addParameter(new FunctionParameter("int", new Variable("value")));
  f->addStatement(new VarDecl("loss",
                              TYPES::BOOL,
                              new LogicalExpr(
                                  new Variable("value"),
                                  OpSymb::smaller,
                                  new BinaryExpr(
                                      new Variable("value"),
                                      OpSymb::addition,
                                      new LiteralInt(1234)))));
  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);
  EXPECT_EQ(mdv.getMaxDepth(), 0);
}

TEST(MultDepthVisitorTests, LogicalAndInReturnStatement) { // NOLINT
  Ast ast;
  AstTestingGenerator::generateAst(16, ast);

  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);
  EXPECT_EQ(mdv.getMaxDepth(), 3);
}

TEST(MultDepthVisitorTests, BinaryExprInReturnStatement) { // NOLINT
  Ast ast;
  AstTestingGenerator::generateAst(17, ast);

  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);
  EXPECT_EQ(mdv.getMaxDepth(), 4);
}

TEST(MultDepthVisitorTests, CallExternalNotSupported) { // NOLINT
  Ast ast;
  AstTestingGenerator::generateAst(3, ast);

  // calculate multiplicative depth
  MultDepthVisitor mdv;

  EXPECT_THROW(mdv.visit(ast), std::logic_error);
}

TEST(MultDepthVisitorTests, IfNotSupported) { // NOLINT
  Ast ast;
  AstTestingGenerator::generateAst(4, ast);

  // calculate multiplicative depth
  MultDepthVisitor mdv;

  EXPECT_THROW(mdv.visit(ast), std::logic_error);
}

TEST(MultDepthVisitorTests, WhileNotSupported) { // NOLINT
  Ast ast;
  AstTestingGenerator::generateAst(11, ast);

  // calculate multiplicative depth
  MultDepthVisitor mdv;

  EXPECT_THROW(mdv.visit(ast), std::logic_error);
}

