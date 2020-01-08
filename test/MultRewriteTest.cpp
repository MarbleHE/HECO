#include <gtest/gtest.h>
#include <Ast.h>
#include <include/visitor/MultRewriteVisitor.h>
#include <include/visitor/PrintVisitor.h>
#include <Operator.h>
#include <include/utilities/TestUtils .h>
#include "AstTestingGenerator.h"
#include "Function.h"
#include "BinaryExpr.h"
#include "VarAssignm.h"

/// Check to ensure that the AST testing generator works as expected.
TEST(MultRewriteTest, astTestingGeneratorTest) { /* NOLINT */
  const int HIGHEST_CASE_NUM = AstTestingGenerator::getLargestId();
  int i = 1;
  for (; i < HIGHEST_CASE_NUM; i++) {
    Ast ast;
    AstTestingGenerator::generateAst(i, ast);
    EXPECT_NE(ast.getRootNode(), nullptr);
  }
  Ast ast;
  EXPECT_THROW(AstTestingGenerator::generateAst(i + 1, ast), std::logic_error);
}

/// Case where multiplication happens in subsequent statements:
///     int prod = inputA * inputB;
///     prod = prod * inputC;
/// [Expected] Rewriting is performed:
///     int prod = inputC * inputB;
///     prod = prod * inputA;
TEST(MultRewriteTest, rewriteSuccessfulSubsequentStatementsMultiplication) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(2, ast);

  // perform rewriting
  MultRewriteVisitor mrv;
  mrv.visit(ast);
  EXPECT_EQ(mrv.getNumChanges(), 1);

  // check presence of expected changes

  //  int prod = inputC * inputB;
  auto func = dynamic_cast<Function*>(ast.getRootNode());
  auto prodDecl = dynamic_cast<VarDecl*>(func->getBody().at(0));
  auto expectedProdDecl = new VarDecl("prod", "int",
                                      new BinaryExpr(
                                          new Variable("inputC"),
                                          OpSymb::multiplication,
                                          new Variable("inputB")));
  EXPECT_TRUE(prodDecl->isEqual(expectedProdDecl));

  //  prod = prod * inputA;
  auto prodAssignm = dynamic_cast<VarAssignm*>(func->getBody().at(1));
  auto expectedProdAssignm = new VarAssignm("prod", new BinaryExpr(
      new Variable("prod"),
      OpSymb::multiplication,
      new Variable("inputA")));
  EXPECT_TRUE(prodAssignm->isEqual(expectedProdAssignm));
}

/// Case where multiplication happens in a single statement:
///     int prod = [inputA * [inputB * inputC]]
/// [Expected] Rewriting is performed:
///     int prod = [inputC * [inputB * inputA]]
TEST(MultRewriteTest, rewriteSuccessfulSingleStatementMultiplication) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(1, ast);

  // perform rewriting
  MultRewriteVisitor mrv;
  mrv.visit(ast);
  EXPECT_EQ(mrv.getNumChanges(), 1);


  // check presence of expected changes

  //  int prod = [inputC * [inputB * inputA]]
  auto func = dynamic_cast<Function*>(ast.getRootNode());
  auto prodDecl = dynamic_cast<VarDecl*>(func->getBody().at(0));
  auto expectedProdDecl = new VarDecl("prod", "int",
                                      new BinaryExpr(
                                          new Variable("inputC"),
                                          OpSymb::multiplication,
                                          new BinaryExpr(
                                              new Variable("inputB"),
                                              OpSymb::multiplication,
                                              new Variable("inputA"))));

  EXPECT_TRUE(prodDecl->isEqual(expectedProdDecl));
}

/// Case where multiplications happen subsequent but there is a statement in between:
///     int prod = inputA * inputB;
///     int rInt = rand();
///     prod = prod * inputC;
/// [Expected] No rewriting is performed.
TEST(MultRewriteTest, noRewriteIfStatementInBetween) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(3, ast);

  // perform rewriting
  MultRewriteVisitor mrv;
  mrv.visit(ast);

  EXPECT_EQ(mrv.changedAst(), false);
}

/// Case where multiplications do happen subsequent but prior statement is in other scope:
///     int prod = inputA * inputB;
///     if (prod > 42) {
///         prod = prod * inputC;
///     }
/// [Expected] No rewriting is performed.
TEST(MultRewriteTest, noRewriteIfOutOfScope) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(4, ast);

  // perform rewriting
  MultRewriteVisitor mrv;
  mrv.visit(ast);

  EXPECT_EQ(mrv.changedAst(), false);
}

/// Case where multiplications do happen subsequent but variable from first statement is not involved in second
/// statement:
///     int prod = inputA * inputB;
///     argPow2 = inputC * inputC;
/// [Expected] No rewriting is performed.
TEST(MultRewriteTest, noRewriteForIndependentStatements) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(5, ast);

  // perform rewriting
  MultRewriteVisitor mrv;
  mrv.visit(ast);

  EXPECT_EQ(mrv.changedAst(), false);
}

/// Case where multiplications do happen subsequent with other variable as assignment target:
///     int prod = inputA * inputB;
///     int prod2 = prod * inputC;
/// [Expected] Rewriting is performed:
///     int prod = inputC * inputB;
///     int prod2 = prod * inputA;
TEST(MultRewriteTest, rewriteNotApplicable) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(6, ast);

  // perform rewriting
  MultRewriteVisitor mrv;
  mrv.visit(ast);
  EXPECT_EQ(mrv.getNumChanges(), 1);

  //  int prod = inputC * inputB;
  auto func = dynamic_cast<Function*>(ast.getRootNode());
  auto prodDecl = dynamic_cast<VarDecl*>(func->getBody().at(0));
  auto expectedProdDecl = new VarDecl("prod", "int",
                                      new BinaryExpr(
                                          new Variable("inputC"),
                                          OpSymb::multiplication,
                                          new Variable("inputB")));
  EXPECT_TRUE(prodDecl->isEqual(expectedProdDecl));

  //  int prod2 = prod * inputA;
  prodDecl = dynamic_cast<VarDecl*>(func->getBody().at(1));
  expectedProdDecl = new VarDecl("prod2", "int", new BinaryExpr(
      new Variable("prod"),
      OpSymb::multiplication,
      new Variable("inputA")));
  EXPECT_TRUE(prodDecl->isEqual(expectedProdDecl));
}

TEST(MultRewriteTest, rewriteSuccessfulSingleStatementMultiplication_EquivTest) { /* NOLINT */
  // create ASTs
  Ast unmodifiedAst;
  AstTestingGenerator::generateAst(1, unmodifiedAst);
  Ast rewrittenAst;
  AstTestingGenerator::generateAst(1, rewrittenAst);
  MultRewriteVisitor().visit(rewrittenAst);

  // create params for eval
  std::map<std::string, Literal*> params = {{"inputA", new LiteralInt(12)},
                                            {"inputB", new LiteralInt(25)},
                                            {"inputC", new LiteralInt(37)}
  };

  // run tests
  astOutputComparer(unmodifiedAst, rewrittenAst, 352734853, 30, params);
}

TEST(MultRewriteTest, noRewriteIfStatementInBetween_EquivTest) { /* NOLINT */
  // create ASTs
  Ast unmodifiedAst;
  AstTestingGenerator::generateAst(2, unmodifiedAst);
  Ast rewrittenAst;
  AstTestingGenerator::generateAst(2, rewrittenAst);
  MultRewriteVisitor().visit(rewrittenAst);

  // create params for eval
  std::map<std::string, Literal*> params = {{"inputA", new LiteralInt(12)},
                                            {"inputB", new LiteralInt(25)},
                                            {"inputC", new LiteralInt(37)}
  };

  // run tests
  astOutputComparer(unmodifiedAst, rewrittenAst, 2341156, 30, params);
}

TEST(MultRewriteTest, noRewriteIfOutOfScope_EquivTest) { /* NOLINT */
  // create ASTs
  Ast unmodifiedAst;
  AstTestingGenerator::generateAst(4, unmodifiedAst);
  Ast rewrittenAst;
  AstTestingGenerator::generateAst(4, rewrittenAst);
  MultRewriteVisitor().visit(rewrittenAst);

  // create params for eval
  std::map<std::string, Literal*> params = {{"inputA", new LiteralInt(12)},
                                            {"inputB", new LiteralInt(25)},
                                            {"inputC", new LiteralInt(37)}
  };

  // run tests
  astOutputComparer(unmodifiedAst, rewrittenAst, 678653456, 30, params);
}

TEST(MultRewriteTest, noRewriteForIndependentStatements_EquivTest) { /* NOLINT */
  // create ASTs
  Ast unmodifiedAst;
  AstTestingGenerator::generateAst(5, unmodifiedAst);
  Ast rewrittenAst;
  AstTestingGenerator::generateAst(5, rewrittenAst);
  MultRewriteVisitor().visit(rewrittenAst);

  // create params for eval
  std::map<std::string, Literal*> params = {{"inputA", new LiteralInt(12)},
                                            {"inputB", new LiteralInt(25)},
                                            {"inputC", new LiteralInt(37)}
  };

  // run tests
  astOutputComparer(unmodifiedAst, rewrittenAst, 87237482, 30, params);
}

TEST(MultRewriteTest, rewriteNotApplicable_EquivTest) { /* NOLINT */
  // create ASTs
  Ast unmodifiedAst;
  AstTestingGenerator::generateAst(6, unmodifiedAst);
  Ast rewrittenAst;
  AstTestingGenerator::generateAst(6, rewrittenAst);
  MultRewriteVisitor().visit(rewrittenAst);

  // create params for eval
  std::map<std::string, Literal*> params = {{"inputA", new LiteralInt(111)},
                                            {"inputB", new LiteralInt(455)},
                                            {"inputC", new LiteralInt(3447)}
  };

  // run tests
  astOutputComparer(unmodifiedAst, rewrittenAst, 87237482, 30, params);
}

