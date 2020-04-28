#include "ast_opt/visitor/MultDepthVisitor.h"
#include "ast_opt/ast/Ast.h"
#include "ast_opt/ast/Operator.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/ArithmeticExpr.h"
#include "ast_opt/ast/LogicalExpr.h"
#include "ast_opt/ast/VarAssignm.h"
#include "AstTestingGenerator.h"
#include "gtest/gtest.h"

TEST(MultDepthVisitorTests, SingleStatementMultiplication) { // NOLINT
  // construct AST
  Ast ast;
  // void f() { int abc = 22 * 32; }
  auto *f = new Function("f");
  f->addStatement(new VarDecl("abc", Types::INT,
                              new ArithmeticExpr(
                                  new LiteralInt(22),
                                  ArithmeticOp::MULTIPLICATION,
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
                                  ArithmeticOp::MULTIPLICATION,
                                  new ArithmeticExpr(
                                      new LiteralInt(32),
                                      ArithmeticOp::MULTIPLICATION,
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
      ArithmeticOp::MULTIPLICATION,
      new Variable("num"))));

  f->addStatement(new VarDecl("beta",
                              Types::INT,
                              new ArithmeticExpr(
                                  new Variable("alpha"),
                                  ArithmeticOp::MULTIPLICATION,
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
                                  LogCompOp::LOGICAL_AND,
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
                                  LogCompOp::LOGICAL_AND,
                                  new LogicalExpr(
                                      new LiteralBool(true),
                                      LogCompOp::LOGICAL_AND,
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
                                  LogCompOp::LOGICAL_AND,
                                  new LiteralBool(false))));

  f->addStatement(new VarDecl("beta", Types::BOOL,
                              new LogicalExpr(
                                  new Variable("alpha"),
                                  LogCompOp::LOGICAL_AND,
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
                                  LogCompOp::SMALLER,
                                  new ArithmeticExpr(
                                      new Variable("value"),
                                      ArithmeticOp::ADDITION,
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
