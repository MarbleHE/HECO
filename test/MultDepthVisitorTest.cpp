#include <gtest/gtest.h>
#include "Ast.h"
#include "MultDepthVisitor.h"
#include "Operator.h"
#include "Function.h"
#include "ArithmeticExpr.h"
#include "LogicalExpr.h"
#include "VarAssignm.h"
#include "AstTestingGenerator.h"

TEST(MultDepthVisitorTests, SingleStatementMultiplication) { // NOLINT
  // construct AST
  Ast ast;
  // void f() { int abc = 22 * 32; }
  auto *f = new Function("f");
  f->addStatement(new VarDecl("abc", Types::INT,
                              new ArithmeticExpr(
                                  new LiteralInt(22),
                                  ArithmeticOp::multiplication,
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
  auto *f = new Function("f");
  f->addStatement(new VarDecl("abc",
                              Types::INT,
                              new ArithmeticExpr(
                                  new LiteralInt(22),
                                  ArithmeticOp::multiplication,
                                  new ArithmeticExpr(
                                      new LiteralInt(32),
                                      ArithmeticOp::multiplication,
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
  auto *f = new Function("f");
  ast.setRootNode(f);
  f->addParameter(new FunctionParameter("int", new Variable("num")));

  f->addStatement(new VarDecl("alpha", Types::INT, new ArithmeticExpr(
      new LiteralInt(32),
      ArithmeticOp::multiplication,
      new Variable("num"))));

  f->addStatement(new VarDecl("beta",
                              Types::INT,
                              new ArithmeticExpr(
                                  new Variable("alpha"),
                                  ArithmeticOp::multiplication,
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
  auto *f = new Function("f");
  f->addStatement(new VarDecl("abc", Types::INT,
                              new LogicalExpr(
                                  new LiteralBool(true),
                                  LogCompOp::logicalAnd,
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
  auto *f = new Function("f");
  f->addStatement(new VarDecl("abc",
                              Types::BOOL,
                              new LogicalExpr(
                                  new LiteralBool(false),
                                  LogCompOp::logicalAnd,
                                  new LogicalExpr(
                                      new LiteralBool(true),
                                      LogCompOp::logicalAnd,
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
  auto *f = new Function("f");
  ast.setRootNode(f);
  f->addParameter(new FunctionParameter("int", new Variable("num")));

  f->addStatement(new VarDecl("alpha", Types::BOOL,
                              new LogicalExpr(
                                  new LiteralBool(true),
                                  LogCompOp::logicalAnd,
                                  new LiteralBool(false))));

  f->addStatement(new VarDecl("beta", Types::BOOL,
                              new LogicalExpr(
                                  new Variable("alpha"),
                                  LogCompOp::logicalAnd,
                                  new LiteralBool(true))));

  // calculate multiplicative depth
  MultDepthVisitor mdv;
  mdv.visit(ast);

  EXPECT_EQ(mdv.getMaxDepth(), 2);
}

TEST(MultDepthVisitorTests, NoLogicalAndOrMultiplicationPresent) { // NOLINT
  Ast ast;
  auto *f = new Function("f");
  ast.setRootNode(f);
  f->addParameter(new FunctionParameter("int", new Variable("value")));
  f->addStatement(new VarDecl("loss",
                              Types::BOOL,
                              new LogicalExpr(
                                  new Variable("value"),
                                  LogCompOp::smaller,
                                  new ArithmeticExpr(
                                      new Variable("value"),
                                      ArithmeticOp::addition,
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

TEST(MultDepthVisitorTests, ArithmeticExprInReturnStatement) { // NOLINT
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

