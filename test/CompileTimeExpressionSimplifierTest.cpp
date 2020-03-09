#include "DotPrinter.h"
#include "CompileTimeExpressionSimplifier.h"
#include "OpSymbEnum.h"
#include "Ast.h"
#include "LiteralInt.h"
#include "LiteralFloat.h"
#include "Return.h"
#include "ArithmeticExpr.h"
#include "VarAssignm.h"
#include "Function.h"
#include "VarDecl.h"
#include "AbstractLiteral.h"
#include "gtest/gtest.h"
#include "Block.h"
#include "UnaryExpr.h"
#include "Call.h"
#include "If.h"
#include "While.h"
#include "AstTestingGenerator.h"
#include "Matrix.h"

class CompileTimeExpressionSimplifierFixture : public ::testing::Test {
 protected:
  Ast ast;
  CompileTimeExpressionSimplifier ctes;
  CompileTimeExpressionSimplifierFixture() = default;

  AbstractExpr *getVariableValue(const std::string &varIdentifier) {
    try {
      return ctes.variableValues.at(varIdentifier);
    } catch (std::out_of_range &outOfRangeException) {
      throw std::logic_error("Variable identifier '" + varIdentifier + "' not found!");
    }
  }
};

TEST_F(CompileTimeExpressionSimplifierFixture, arithmeticExpr_literalsOnly_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //  plaintext_int alpha = 22 * 11;
  // }
  auto function = new Function("compute");
  auto arithmeticExpr = new ArithmeticExpr(
      new LiteralInt(22),
      ArithmeticOp::multiplication,
      new LiteralInt(11));
  auto varAssignm = new VarDecl("alpha",
                                new Datatype(Types::INT, false),
                                arithmeticExpr);

  // connect objects
  function->addStatement(varAssignm);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
  auto alphaValue = getVariableValue("alpha");
  EXPECT_EQ(alphaValue->castTo<LiteralInt>()->getValue(), 242);
  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);
  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, arithmeticExpr_variableUnknown_rhsOperandEvaluableOnly) { /* NOLINT */
  // void compute(encrypted_int encryptedA) {
  //  plaintext_int alpha = encryptedA * (4*7);
  // }
  auto function = new Function("compute");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(new Datatype(Types::INT, true),
                             new Variable("encryptedA"))});
  auto arithmeticExpr = new ArithmeticExpr(
      new Variable("encryptedA"),
      ArithmeticOp::multiplication,
      new ArithmeticExpr(new LiteralInt(4),
                         ArithmeticOp::multiplication,
                         new LiteralInt(7)));
  auto varAssignm = new VarDecl("alpha", new Datatype(Types::INT, false), arithmeticExpr);

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varAssignm);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that value of alpha  variable
  EXPECT_EQ(ctes.variableValues.size(), 1);
  // check that the rhs operand of arithmeticExpr is simplified
  auto expected = new ArithmeticExpr(new Variable("encryptedA"),
                                     ArithmeticOp::multiplication,
                                     new LiteralInt(28));
  EXPECT_TRUE(getVariableValue("alpha")->isEqual(expected));
  EXPECT_EQ(function->getBodyStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, arithmeticExpr_variableKnown_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //   plaintext_int parameterA = 43;
  //   plaintext_int alpha = parameterA * (4*7);
  // }
  auto function = new Function("compute");
  auto varDeclParameterA = new VarDecl("parameterA", 43);
  auto arithmeticExpr = new ArithmeticExpr(
      new Variable("parameterA"),
      ArithmeticOp::multiplication,
      new ArithmeticExpr(new LiteralInt(4),
                         ArithmeticOp::multiplication,
                         new LiteralInt(7)));
  auto varDeclAlpha = new VarDecl("alpha",
                                  new Datatype(Types::INT, false),
                                  arithmeticExpr);

  // connect objects
  function->addStatement(varDeclParameterA);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
  auto alphaValue = getVariableValue("alpha");
  EXPECT_EQ(alphaValue->castTo<LiteralInt>()->getValue(), 1'204);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, arithmeticExpr_variablesUnknown_notAnythingEvaluable) { /* NOLINT */
  // void compute(encrypted_int encryptedA, plaintext_int plaintextB) {
  //  plaintext_int alpha = encryptedA * (4*plaintextB);
  // }
  auto function = new Function("compute");
  auto functionParameters =
      std::vector<FunctionParameter *>(
          {
              new FunctionParameter(new Datatype(Types::INT, true),
                                    new Variable("encryptedA")),
              new FunctionParameter(new Datatype(Types::INT, false),
                                    new Variable("plaintextB"))});
  function->setParameterList(new ParameterList(functionParameters));

  auto arithmeticExpr = new ArithmeticExpr(
      new Variable("encryptedA"),
      ArithmeticOp::multiplication,
      new ArithmeticExpr(new LiteralInt(4),
                         ArithmeticOp::multiplication,
                         new Variable("plaintextB")));
  auto varDeclAlpha = new VarDecl("alpha",
                                  new Datatype(Types::INT, false),
                                  arithmeticExpr);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);
  auto numberOfNodesBeforeSimplification = ast.getAllNodes().size();

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
  auto expected = new ArithmeticExpr(
      new Variable("encryptedA"),
      ArithmeticOp::multiplication,
      new ArithmeticExpr(new LiteralInt(4),
                         ArithmeticOp::multiplication,
                         new Variable("plaintextB")));
  EXPECT_TRUE(getVariableValue("alpha")->isEqual(expected));

  // check that 9 nodes were deleted and the function's body is empty
  EXPECT_EQ(numberOfNodesBeforeSimplification - 9, ast.getAllNodes().size());
  EXPECT_EQ(function->getBodyStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, logicalExpr_literalsOnly_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //  plaintext_bool alpha = true && false;
  // }
  auto function = new Function("compute");
  auto logicalExpr = new LogicalExpr(
      new LiteralBool(true),
      LogCompOp::logicalAnd,
      new LiteralBool(false));
  auto varAssignm = new VarDecl("alpha",
                                new Datatype(Types::BOOL, false),
                                logicalExpr);

  // connect objects
  function->addStatement(varAssignm);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
  auto alphaValue = getVariableValue("alpha");
  EXPECT_EQ(alphaValue->castTo<LiteralBool>()->getValue(), false);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, logicalExpr_variableUnknown_lhsOperandEvaluableOnly) { /* NOLINT */
  //  -- input --
  // void compute(encrypted_bool encryptedA) {
  //  plaintext_bool alpha = (true ^ false) || encryptedA;
  // }
  //  -- expected --
  // variableValues['alpha'] = true
  auto function = new Function("compute");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(new Datatype(Types::BOOL, true),
                             new Variable("encryptedA"))});
  auto logicalExpr = new LogicalExpr(
      new LogicalExpr(new LiteralBool(true),
                      LogCompOp::logicalXor,
                      new LiteralBool(false)),
      LogCompOp::logicalOr,
      new Variable("encryptedA"));
  auto varAssignm = new VarDecl("alpha",
                                new Datatype(Types::BOOL, false),
                                logicalExpr);

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varAssignm);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that alpha's lhs operand is computed
  auto expected = new LiteralBool(true);
  EXPECT_TRUE(getVariableValue("alpha")->isEqual(expected));
  // check that the variable declaration statement is deleted
  EXPECT_EQ(function->getBodyStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, logicalExpr_variableKnown_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //   plaintext_bool parameterA = true;
  //   plaintext_bool alpha = parameterA || (false && true);
  // }
  auto function = new Function("compute");
  auto varDeclParameterA = new VarDecl("parameterA", true);
  auto logicalExpr = new LogicalExpr(
      new Variable("parameterA"),
      LogCompOp::logicalOr,
      new LogicalExpr(new LiteralBool(false),
                      LogCompOp::logicalAnd,
                      new LiteralBool(true)));
  auto varDeclAlpha = new VarDecl("alpha",
                                  new Datatype(Types::BOOL, false),
                                  logicalExpr);

  // connect objects
  function->addStatement(varDeclParameterA);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
  auto alphaValue = getVariableValue("alpha");
  EXPECT_EQ(alphaValue->castTo<LiteralBool>()->getValue(), true);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, logicalExpr_variablesUnknown_notAnythingEvaluable) { /* NOLINT */
  //  -- input --
  // void compute(encrypted_bool encryptedA, plaintext_bool paramB) {
  //  plaintext_bool alpha = encryptedA && (true ^ encryptedB);
  // }
  //  -- expected --
  // variableValues['alpha'] = encryptedA && !encryptedB
  auto function = new Function("compute");
  auto functionParameters = std::vector<FunctionParameter *>(
      {new FunctionParameter(new Datatype(Types::BOOL, true),
                             new Variable("encryptedA")),
       new FunctionParameter(new Datatype(Types::BOOL, false),
                             new Variable("paramB"))});
  function->setParameterList(new ParameterList(functionParameters));

  auto logicalExpr = new LogicalExpr(
      new Variable("encryptedA"),
      LogCompOp::logicalAnd,
      new LogicalExpr(new LiteralBool(true),
                      LogCompOp::logicalXor,
                      new Variable("encryptedB")));
  auto varDeclAlpha = new VarDecl("alpha",
                                  new Datatype(Types::BOOL, false),
                                  logicalExpr);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);
  auto numberOfNodesBeforeSimplification = ast.getAllNodes().size();

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly (could not be computed but is saved)
  auto expected = new LogicalExpr(
      new Variable("encryptedA"),
      LogCompOp::logicalAnd,
      new UnaryExpr(negation, new Variable("encryptedB")));
  EXPECT_TRUE(getVariableValue("alpha")->isEqual(expected));

  // check that 9 nodes are deleted
  EXPECT_EQ(numberOfNodesBeforeSimplification - 9, ast.getAllNodes().size());

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, unaryExpr_literalsOnly_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //  plaintext_bool truthValue = !false;
  // }
  auto function = new Function("compute");
  auto unaryExpr = new UnaryExpr(UnaryOp::negation, new LiteralBool(false));
  auto varAssignm = new VarDecl("truthValue",
                                new Datatype(Types::BOOL, false),
                                unaryExpr);

  // connect objects
  function->addStatement(varAssignm);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'truthValue' is computed correctly
  auto alphaValue = getVariableValue("truthValue");
  EXPECT_EQ(alphaValue->castTo<LiteralBool>()->getValue(), true);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, unaryExpr_variableKnown_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //  plaintext_bool alpha = true;
  //  plaintext_bool beta = !alpha;
  // }
  auto function = new Function("compute");
  auto varDeclParameterA = new VarDecl("alpha", true);
  auto unaryExpr = new UnaryExpr(UnaryOp::negation, new Variable("alpha"));
  auto varDeclAlpha = new VarDecl("beta",
                                  new Datatype(Types::BOOL, false),
                                  unaryExpr);

  // connect objects
  function->addStatement(varDeclParameterA);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is stored correctly
  auto alphaValue = getVariableValue("alpha");
  EXPECT_EQ(alphaValue->castTo<LiteralBool>()->getValue(), true);

  // check that 'beta' is computed correctly
  auto betaValue = getVariableValue("beta");
  EXPECT_EQ(betaValue->castTo<LiteralBool>()->getValue(), false);

  // check that both statements and their children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, unaryExpr_variableUnknown_notEvaluable) { /* NOLINT */
  // void compute(plaintext_bool paramA) {
  //  plaintext_bool beta = !paramA;
  // }
  auto function = new Function("compute");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(new Datatype(Types::BOOL, false),
                             new Variable("paramA"))});
  auto unaryExpr = new UnaryExpr(UnaryOp::negation, new Variable("paramA"));
  auto varDeclAlpha = new VarDecl("beta",
                                  new Datatype(Types::BOOL, false),
                                  unaryExpr);

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that beta was computed
  EXPECT_EQ(ctes.variableValues.size(), 1);
  EXPECT_TRUE(getVariableValue("beta")->isEqual(new UnaryExpr(UnaryOp::negation, new Variable("paramA"))));
  // check that statements is deleted
  EXPECT_EQ(function->getBodyStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, varAssignm_variablesKnown_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //  plaintext_float alpha = 1.23;
  //  alpha = 2.75;
  //  plaintext_float beta = alpha;
  // }
  auto function = new Function("compute");
  auto varDeclAlpha = new VarDecl("alpha", 1.23f);
  auto varAssignmAlpha = new VarAssignm("alpha", new LiteralFloat(2.75));
  auto varDeclBeta = new VarDecl("beta",
                                 new Datatype(Types::FLOAT), new Variable("alpha"));

  // connect objects
  function->addStatement(varDeclAlpha);
  function->addStatement(varAssignmAlpha);
  function->addStatement(varDeclBeta);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is stored correctly
  auto alphaValue = getVariableValue("alpha");
  EXPECT_EQ(alphaValue->castTo<LiteralFloat>()->getValue(), 2.75);

  // check that 'beta' is assigned correctly
  auto betaValue = getVariableValue("beta");
  EXPECT_EQ(betaValue->castTo<LiteralFloat>()->getValue(), 2.75);

  // check that the statements and their children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, varAssignm_previouslyDeclaredNonInitializedVariable) { /* NOLINT */
  // void compute() {
  //  plaintext_float alpha;
  //  alpha = 2.95;
  // }
  auto function = new Function("compute");
  auto varDeclAlpha = new VarDecl("alpha", Types::FLOAT, nullptr);
  auto varAssignmAlpha = new VarAssignm("alpha", new LiteralFloat(2.95));

  // connect objects
  function->addStatement(varDeclAlpha);
  function->addStatement(varAssignmAlpha);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
  auto alphaValue = getVariableValue("alpha");
  EXPECT_EQ(alphaValue->castTo<LiteralFloat>()->getValue(), 2.95f);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture,  /* NOLINT */
       varAssignm_variableDeclarationOnly_correctInitialFloatValueExpected) {
  // void compute() {
  //  plaintext_float alpha;
  // }
  auto function = new Function("compute");
  auto varDeclAlpha = new VarDecl("alpha", Types::FLOAT, nullptr);

  // connect objects
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' has correct initial value
  EXPECT_EQ(getVariableValue("alpha")->castTo<LiteralFloat>()->getValue(), 0.0f);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, varAssignm_assignmentToParameter) { /* NOLINT */
  // void compute(plaintext_float alpha) {
  //  alpha = 42.24;
  // }
  auto function = new Function("compute");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(new Datatype(Types::FLOAT, false),
                             new Variable("alpha"))});
  auto varAssignmAlpha = new VarAssignm("alpha", new LiteralFloat(42.24));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varAssignmAlpha);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
  auto alphaValue = getVariableValue("alpha");
  EXPECT_EQ(alphaValue->castTo<LiteralFloat>()->getValue(), 42.24f);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture,  /* NOLINT */
       varAssignm_symbolicTerms_circularDependency) {
  //  -- input --
  // int Foo(plaintext_int x, plaintext_int y) {
  //  x = y+3
  //  y = x+2
  //  return x+y
  // }
  //  -- expected --
  // int Foo(plaintext_int x, plaintext_int y) {
  //  return y+y+8
  // }
  auto function = new Function("Foo");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(
          new Datatype(Types::INT, false), new Variable("x")),
       new FunctionParameter(
           new Datatype(Types::INT, false), new Variable("y"))});
  auto varAssignmX = new VarAssignm("x",
                                    new ArithmeticExpr(
                                        new Variable("y"),
                                        ArithmeticOp::addition,
                                        3));
  auto varAssignmY = new VarAssignm("y",
                                    new ArithmeticExpr(
                                        new Variable("x"),
                                        ArithmeticOp::addition,
                                        2));
  auto returnStmt = new Return(
      new ArithmeticExpr(
          new Variable("x"),
          ArithmeticOp::addition,
          new Variable("y")));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varAssignmX);
  function->addStatement(varAssignmY);
  function->addStatement(returnStmt);
  ast.setRootNode(function);
  auto originalNumberOfNodes = ast.getAllNodes().size();

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that the number of nodes decreased
  EXPECT_TRUE(ast.getAllNodes().size() < originalNumberOfNodes);

  // check that simplification generated the expected simplified AST
  auto expectedAst = new ArithmeticExpr(
      new ArithmeticExpr(
          new LiteralInt(8),
          ArithmeticOp::addition,
          new Variable("y")),
      ArithmeticOp::addition,
      new Variable("y"));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedAst));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, return_literalOnly_expectedNoChange) { /* NOLINT */
  // float compute() {
  //  return 42.24;
  // }
  // -- expected –-
  // no change as cannot be simplified further
  auto function = new Function("compute");
  auto returnStatement = new Return(new LiteralFloat(42.24f));

  // connect objects
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that none of the nodes are deleted
  auto numberOfNodesBeforeSimplification = ast.getAllNodes().size();
  EXPECT_EQ(numberOfNodesBeforeSimplification, ast.getAllNodes().size());

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture,  /* NOLINT */
       return_variableKnown_expectedSubstitutionAndStatementDeletion) {
  // int compute() {
  //  int b = 23;
  //  return b;
  // }
  // -- expected --
  // int compute() {
  //  return 23;
  // }
  auto function = new Function("compute");
  auto varDeclB = new VarDecl("b", 23);
  auto returnStatement = new Return(new Variable("b"));

  // connect objects
  function->addStatement(varDeclB);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 1);
  ASSERT_EQ(function->getBody()->getStatements().front(), returnStatement);

  // check that the variable 'b' in the Return statement was replaced by b's value
  auto firstReturnExpr = returnStatement->getReturnExpressions().front();
  auto newLiteralIntNode = dynamic_cast<LiteralInt *>(firstReturnExpr);
  ASSERT_NE(newLiteralIntNode, nullptr);
  EXPECT_EQ(newLiteralIntNode->getValue(), 23);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       return_variableAndArithmeticExpressionKnown_expectedLiteralIntReturnValue) {
  // int compute() {
  //  int b = 23;
  //  return b + 99;
  // }
  // -- expected --
  // int compute() {
  //  return 122;
  // }
  auto function = new Function("compute");
  auto varDeclB = new VarDecl("b", 23);
  auto returnStatement = new Return(
      new ArithmeticExpr(new Variable("b"),
                         ArithmeticOp::addition,
                         new LiteralInt(99)));

  // connect objects
  function->addStatement(varDeclB);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 1);
  ASSERT_EQ(function->getBody()->getStatements().front(), returnStatement);

  // check that the expression b+99 was simplified by its value
  auto firstReturnExpr = returnStatement->getReturnExpressions().front();
  auto newLiteralIntNode = dynamic_cast<LiteralInt *>(firstReturnExpr);
  ASSERT_NE(newLiteralIntNode, nullptr);
  EXPECT_EQ(newLiteralIntNode->getValue(), 122);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, return_variableUnknown_expectedNoChange) { /* NOLINT */
  // int compute(plaintext_int b) {
  //  return b + 99;
  // }
  // -- expected --
  // no change
  auto function = new Function("compute");
  auto returnStatement = new Return(
      new ArithmeticExpr(new Variable("b"),
                         ArithmeticOp::addition,
                         new LiteralInt(99)));

  // connect objects
  function->addStatement(returnStatement);
  ast.setRootNode(function);
  auto numberOfNodesBeforeSimplification = ast.getAllNodes().size();

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that none of the nodes are deleted
  EXPECT_EQ(numberOfNodesBeforeSimplification, ast.getAllNodes().size());

  // check that 'b' remains unknown
  EXPECT_THROW(getVariableValue("b"), std::logic_error);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, return_multipleReturnValues_expectedPartlyEvaluation) { /* NOLINT */
  // int,int,int compute(plaintext_int a) {
  //  int b = 3 + 4;
  //  return a*b, 2-b, 21;
  // }
  // -- expected --
  // int,int,int compute(plaintext_int a) {
  //  return a*7, -5, 21;
  // }
  auto function = new Function("compute");
  auto functionParam = new ParameterList(
      {new FunctionParameter(new Datatype(Types::INT), new Variable("a"))});
  auto varDecl = new VarDecl("b",
                             new Datatype(Types::INT),
                             new ArithmeticExpr(
                                 new LiteralInt(3),
                                 ArithmeticOp::addition,
                                 new LiteralInt(4)));
  auto returnStatement =
      new Return({
                     new ArithmeticExpr(
                         new Variable("a"),
                         ArithmeticOp::multiplication,
                         new Variable("b")),
                     new ArithmeticExpr(
                         new LiteralInt(2),
                         ArithmeticOp::subtraction,
                         new Variable("b")),
                     new LiteralInt(21)});

  // connect objects
  function->setParameterList(functionParam);
  function->addStatement(varDecl);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that the Return statement has three return values
  EXPECT_EQ(returnStatement->getReturnExpressions().size(), 3);

  // check return expression 1: a*7
  EXPECT_TRUE(returnStatement->getReturnExpressions().at(0)
                  ->isEqual(new ArithmeticExpr(new Variable("a"), ArithmeticOp::multiplication, new LiteralInt(7))));
  // check return expression 2: -5
  EXPECT_TRUE(returnStatement->getReturnExpressions().at(1)->isEqual(new LiteralInt(-5)));
  // check return expression 3: 21
  EXPECT_TRUE(returnStatement->getReturnExpressions().at(2)->isEqual(new LiteralInt(21)));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       ifStmt_conditionValueIsKnown_thenIsAlwaysExecutedNoElseIsPresent_expectedIfRemoval) {
  //  -- input --
  //  int compute() {
  //    int a = 512;
  //    int b = 22;
  //    if (b+12 > 20) {
  //      a = a*2;
  //    }
  //    return a*32;
  //  }
  //  -- expected --
  //  int compute() {
  //    return 32'768;
  //  }
  auto function = new Function("compute");
  auto varDeclA = new VarDecl("a",
                              new Datatype(Types::INT),
                              new LiteralInt(512));
  auto varDeclB = new VarDecl("b",
                              new Datatype(Types::INT),
                              new LiteralInt(22));
  auto ifStmt = new If(
      new LogicalExpr(
          new ArithmeticExpr(
              new Variable("b"),
              ArithmeticOp::addition,
              new LiteralInt(12)),
          LogCompOp::greater,
          new LiteralInt(20)),
      new VarAssignm("a", new ArithmeticExpr(
          new Variable("a"),
          ArithmeticOp::multiplication,
          new LiteralInt(2))));
  auto returnStatement =
      new Return(new ArithmeticExpr(new Variable("a"),
                                    ArithmeticOp::multiplication,
                                    new LiteralInt(32)));

  // connect objects
  function->addStatement(varDeclA);
  function->addStatement(varDeclB);
  function->addStatement(ifStmt);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is only one statement left
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  // check that the return statement contains exactly one return value
  EXPECT_EQ(returnStatement->getReturnExpressions().size(), 1);
  // check that the return value is the expected one
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(new LiteralInt(32'768)));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       ifStmt_conditionValueIsKnown_thenIsAlwaysExecutedAndElseIsPresent_expectedIfRemoval) {
  //  -- input --
  //  int compute() {
  //    int a = 512;
  //    int b = 22;
  //    if (b+12 > 20) {
  //      a = a*2;
  //    } else {
  //      a = 1;
  //    }
  //    return a*32;
  //  }
  //  -- expected --
  //  int compute() {
  //    return 32'768;
  //  }
  auto function = new Function("compute");
  auto varDeclA = new VarDecl("a",
                              new Datatype(Types::INT),
                              new LiteralInt(512));
  auto varDeclB = new VarDecl("b",
                              new Datatype(Types::INT),
                              new LiteralInt(22));

  auto condition = new LogicalExpr(
      new ArithmeticExpr(new Variable("b"), ArithmeticOp::addition, new LiteralInt(12)),
      LogCompOp::greater,
      new LiteralInt(20));
  auto thenBranch = new VarAssignm("a", new ArithmeticExpr(
      new Variable("a"),
      ArithmeticOp::multiplication,
      new LiteralInt(2)));
  auto elseBranch = new VarAssignm("a", new LiteralInt(1));
  auto ifStmt = new If(condition, thenBranch, elseBranch);

  auto returnStatement =
      new Return(new ArithmeticExpr(new Variable("a"),
                                    ArithmeticOp::multiplication,
                                    new LiteralInt(32)));

  // connect objects
  function->addStatement(varDeclA);
  function->addStatement(varDeclB);
  function->addStatement(ifStmt);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is only one statement left
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  EXPECT_EQ(function->getBodyStatements().front(), returnStatement);
  // check that the return statement contains exactly one return value
  EXPECT_EQ(returnStatement->getReturnExpressions().size(), 1);
  // check that the return value is the expected one
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(new LiteralInt(32'768)));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       ifStmt_conditionValueIsKnown_elseIsAlwaysExecuted_expectedIfRemoval) {
  //  -- input --
  //  int compute() {
  //    int a = 512;
  //    int b = 22;
  //    if (b+12 < 20) {
  //      a = a*2;
  //    } else {
  //      a = 1;
  //    }
  //    return a*32;
  //  }
  //  -- expected --
  //  int compute() {
  //    return 32;
  //  }
  auto function = new Function("compute");
  auto varDeclA = new VarDecl("a",
                              new Datatype(Types::INT),
                              new LiteralInt(512));
  auto varDeclB = new VarDecl("b",
                              new Datatype(Types::INT),
                              new LiteralInt(22));

  auto condition = new LogicalExpr(
      new ArithmeticExpr(new Variable("b"), ArithmeticOp::addition, new LiteralInt(12)),
      LogCompOp::smaller,
      new LiteralInt(20));
  auto thenBranch = new VarAssignm("a", new ArithmeticExpr(
      new Variable("a"),
      ArithmeticOp::multiplication,
      new LiteralInt(2)));
  auto elseBranch = new VarAssignm("a", new LiteralInt(1));
  auto ifStmt = new If(condition, thenBranch, elseBranch);

  auto returnStatement =
      new Return(new ArithmeticExpr(new Variable("a"),
                                    ArithmeticOp::multiplication,
                                    new LiteralInt(32)));

  // connect objects
  function->addStatement(varDeclA);
  function->addStatement(varDeclB);
  function->addStatement(ifStmt);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'a' was memorized correctly
  EXPECT_EQ(getVariableValue("a")->castTo<LiteralInt>()->getValue(), 1);
  // check that there is only one statement left
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  EXPECT_EQ(function->getBodyStatements().front(), returnStatement);
  // check that the return statement contains exactly one return value
  EXPECT_EQ(returnStatement->getReturnExpressions().size(), 1);
  // check that the return value is the expected one
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(new LiteralInt(32)));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       ifStmt_conditionValueIsUnknown_thenBranchOnlyExists_thenBranchEvaluable_expectedRewriting) {
  //  -- input --
  //  int compute(plaintext_int a) {
  //    int b = 22;
  //    if (a > 20) {
  //      b = 2*b;
  //    }
  //    return b;
  //  }
  //  -- expected --
  //  int compute() {
  //    return [a>20]*44+[1-[a>20]]*22;
  //  }
  auto function = new Function("compute");
  auto functionParameter = new ParameterList({
                                                 new FunctionParameter(new Datatype(Types::INT),
                                                                       new Variable("a"))});
  auto varDeclB = new VarDecl("b",
                              new Datatype(Types::INT),
                              new LiteralInt(22));
  auto ifStmtCondition = new LogicalExpr(new Variable("a"),
                                         LogCompOp::greater,
                                         new LiteralInt(20));
  auto ifStmt = new If(
      ifStmtCondition,
      new VarAssignm("b", new ArithmeticExpr(
          new LiteralInt(2),
          ArithmeticOp::multiplication,
          new Variable("b"))));
  auto returnStatement =
      new Return(new Variable("b"));

  // connect objects
  function->setParameterList(functionParameter);
  function->addStatement(varDeclB);
  function->addStatement(ifStmt);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is only one statement left
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  // check that the return statement contains exactly one return value
  EXPECT_EQ(returnStatement->getReturnExpressions().size(), 1);
  // check that the return value is the expected one
  auto expectedResult = new ArithmeticExpr(
      new ArithmeticExpr(
          new LogicalExpr(new Variable("a"),
                          LogCompOp::greater,
                          new LiteralInt(20)),
          ArithmeticOp::multiplication,
          new LiteralInt(44)),
      ArithmeticOp::addition,
      new ArithmeticExpr(
          new ArithmeticExpr(
              new LiteralInt(1),
              ArithmeticOp::subtraction,
              new LogicalExpr(new Variable("a"),
                              LogCompOp::greater,
                              new LiteralInt(20))),
          ArithmeticOp::multiplication,
          new LiteralInt(22)));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedResult));
  // check that 'b' was memorized correctly
  EXPECT_TRUE(getVariableValue("b")->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       ifStmt_conditionValueIsUnknown_thenBranchOnlyExists_expectedRemovalOfElseClauseInResultBecauseVariableBIsNull) {
  //  -- input --
  //  int compute(plaintext_int a) {
  //    plaintext_int b;  // implicit: b=0
  //    if (a > 20) {
  //      plaintext_int c = 642;
  //      b = 2*c-1;
  //    }
  //    return b;
  //  }
  //  -- expected --
  //  int compute() {
  //    return [a>20]*1'283;
  //  }
  auto function = new Function("compute");
  auto functionParameter = new ParameterList({
                                                 new FunctionParameter(new Datatype(Types::INT),
                                                                       new Variable("a"))});
  auto varDeclB = new VarDecl("b", new Datatype(Types::INT));
  auto ifStmtCondition = new LogicalExpr(new Variable("a"),
                                         LogCompOp::greater,
                                         new LiteralInt(20));
  auto thenStatements = std::vector<AbstractStatement *>(
      {new VarDecl("c", 642),
       new VarAssignm("b",
                      new ArithmeticExpr(
                          new ArithmeticExpr(
                              new LiteralInt(2),
                              ArithmeticOp::multiplication,
                              new Variable("c")),
                          ArithmeticOp::subtraction,
                          new LiteralInt(1)))});
  auto thenBranch = new Block(thenStatements);
  auto ifStmt = new If(ifStmtCondition, thenBranch);
  auto returnStatement =
      new Return(new Variable("b"));

  // connect objects
  function->setParameterList(functionParameter);
  function->addStatement(varDeclB);
  function->addStatement(ifStmt);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is only one statement left
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  // check that the return statement contains exactly one return value
  EXPECT_EQ(returnStatement->getReturnExpressions().size(), 1);
  // check that the return value is the expected one
  auto expectedResult =
      new ArithmeticExpr(
          new LogicalExpr(new Variable("a"),
                          LogCompOp::greater,
                          new LiteralInt(20)),
          ArithmeticOp::multiplication,
          new LiteralInt(1'283));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedResult));
  // check that variable values were memorized correctly
  EXPECT_TRUE(getVariableValue("b")->isEqual(expectedResult));
  // variable 'c' is not expected to be memorized because it's declared in the Then-branch only
  EXPECT_THROW(getVariableValue("c"), std::logic_error);

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       ifStmt_conditionValueIsUnknown_thenBranchOnlyExists_varDeclInThenBranch_expectedRewritingOfIfStatement) {
  //  -- input --
  //  int compute(plaintext_int a) {
  //    plaintext_int b = 42;
  //    if (a > 20) {
  //      plaintext_int c = 642;
  //      b = 2*c-1;
  //    }
  //    return b;
  //  }
  //  -- expected --
  //  int compute() {
  //    return [a>20]*1'283+[1-[a>20]]*42;
  //  }
  auto function = new Function("compute");
  auto functionParameter = new ParameterList({
                                                 new FunctionParameter(new Datatype(Types::INT),
                                                                       new Variable("a"))});
  auto varDeclB = new VarDecl("b", new Datatype(Types::INT), new LiteralInt(42));
  auto ifStmtCondition = new LogicalExpr(new Variable("a"),
                                         LogCompOp::greater,
                                         new LiteralInt(20));
  auto thenStatements = std::vector<AbstractStatement *>(
      {new VarDecl("c", 642),
       new VarAssignm("b",
                      new ArithmeticExpr(
                          new ArithmeticExpr(
                              new LiteralInt(2),
                              ArithmeticOp::multiplication,
                              new Variable("c")),
                          ArithmeticOp::subtraction,
                          new LiteralInt(1)))});
  auto thenBranch = new Block(thenStatements);
  auto ifStmt = new If(ifStmtCondition, thenBranch);
  auto returnStatement =
      new Return(new Variable("b"));

  // connect objects
  function->setParameterList(functionParameter);
  function->addStatement(varDeclB);
  function->addStatement(ifStmt);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is only one statement left
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  // check that the return statement contains exactly one return value
  EXPECT_EQ(returnStatement->getReturnExpressions().size(), 1);
  // check that the return value is the expected one
  auto expectedResult = new ArithmeticExpr(
      new ArithmeticExpr(
          new LogicalExpr(new Variable("a"),
                          LogCompOp::greater,
                          new LiteralInt(20)),
          ArithmeticOp::multiplication,
          new LiteralInt(1'283)),
      ArithmeticOp::addition,
      new ArithmeticExpr(new ArithmeticExpr(
          new LiteralInt(1),
          ArithmeticOp::subtraction,
          new LogicalExpr(new Variable("a"),
                          LogCompOp::greater,
                          new LiteralInt(20))),
                         ArithmeticOp::multiplication,
                         new LiteralInt(42)));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedResult));
  // check that 'b' was memorized correctly
  EXPECT_TRUE(getVariableValue("b")->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       ifStmt_conditionValueIsUnknown_thenAndElseExists_returnValueIsInputVariable_expectedRewritingOfIfStatement) {
  //  -- input --
  //  int compute(plaintext_int factor, plaintext_int threshold) {
  //    int b;
  //    if (threshold < 11) {
  //      b = 2*factor;
  //    } else {
  //      b = factor;
  //    }
  //    return b;
  //  }
  //  -- expected --
  //  int compute() {
  //    return [threshold<11]*2*factor + [1-[threshold<11]]*factor;
  //  }
  auto function = new Function("compute");
  auto functionParameter = new ParameterList(
      {new FunctionParameter(new Datatype(Types::INT), new Variable("factor")),
       new FunctionParameter(new Datatype(Types::INT), new Variable("threshold"))});
  auto varDecl = new VarDecl("b", new Datatype(Types::INT));
  auto ifStmtCondition = new LogicalExpr(new Variable("threshold"),
                                         LogCompOp::smaller,
                                         new LiteralInt(11));
  auto thenBranch = new VarAssignm("b",
                                   new ArithmeticExpr(
                                       new LiteralInt(2),
                                       ArithmeticOp::multiplication,
                                       new Variable("factor")));
  auto elseBranch = new VarAssignm("b", new Variable("factor"));

  auto ifStmt = new If(ifStmtCondition, thenBranch, elseBranch);
  auto returnStatement =
      new Return(new Variable("b"));

  // connect objects
  function->addStatement(varDecl);
  function->setParameterList(functionParameter);
  function->addStatement(ifStmt);
  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is only one statement left
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  // check that the return statement contains exactly one return value
  EXPECT_EQ(returnStatement->getReturnExpressions().size(), 1);
  // check that the return value is the expected one
  auto expectedResult = new ArithmeticExpr(
      new ArithmeticExpr(
          new LogicalExpr(new Variable("threshold"),
                          LogCompOp::smaller,
                          new LiteralInt(11)),
          ArithmeticOp::multiplication,
          new ArithmeticExpr(new LiteralInt(2), ArithmeticOp::multiplication, new Variable("factor"))),
      ArithmeticOp::addition,
      new ArithmeticExpr(new ArithmeticExpr(
          new LiteralInt(1),
          ArithmeticOp::subtraction,
          new LogicalExpr(new Variable("threshold"),
                          LogCompOp::smaller,
                          new LiteralInt(11))),
                         ArithmeticOp::multiplication,
                         new Variable("factor")));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedResult));
  // check that 'b' was memorized correctly
  EXPECT_TRUE(getVariableValue("b")->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, /* NOLINT */
       ifStmt_nestedIfStatements_expectedRewritingOfBothIfStatement) {
  //  -- input --
  //  int compute(plaintext_int factor, plaintext_int threshold) {
  //    int b = 99;
  //    if (threshold > 11) {
  //        b = b/3;          // b = 33;
  //      if (factor > 9) {
  //        b = b*2*factor;   // b = 33*2*factor = 66*factor;
  //      } else {
  //        b = b*factor;     // b = 33*factor
  //      }
  //    }
  //    return b;
  //  }
  //  -- expected --
  //  int compute() {
  //    return [factor>9]*[threshold>11]*66*factor + [1-[factor>9]]*[threshold>11]*33*factor + [1-[threshold>11]]*99;
  //  }
  auto function = new Function("compute");
  auto functionParameter = new ParameterList(
      {new FunctionParameter(new Datatype(Types::INT), new Variable("factor")),
       new FunctionParameter(new Datatype(Types::INT), new Variable("threshold"))});

  auto varDeclB = new VarDecl("b", 99);

  auto innerIfStatementCondition = new LogicalExpr(new Variable("factor"), LogCompOp::greater, new LiteralInt(9));
  auto innerIfStatement = new If(innerIfStatementCondition,
                                 new VarAssignm("b", new ArithmeticExpr(
                                     new ArithmeticExpr(new Variable("b"), ArithmeticOp::multiplication, 2),
                                     ArithmeticOp::multiplication,
                                     new Variable("factor"))),
                                 new VarAssignm("b",
                                                new ArithmeticExpr(new Variable("b"),
                                                                   ArithmeticOp::multiplication,
                                                                   new Variable("factor"))));

  auto outerIfStmtThenBlock =
      new Block({new VarAssignm("b", new ArithmeticExpr(new Variable("b"), ArithmeticOp::division, new LiteralInt(3))),
                 innerIfStatement});
  auto outerIfStatementCondition = new LogicalExpr(new Variable("threshold"), LogCompOp::greater, new LiteralInt(11));
  auto outerIfStmt = new If(outerIfStatementCondition, outerIfStmtThenBlock);

  auto returnStmt = new Return(new Variable("b"));

  // connect objects
  function->setParameterList(functionParameter);
  function->addStatement(varDeclB);
  function->addStatement(outerIfStmt);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is only one statement left
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  // check that the return statement contains exactly one return value
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  // check that the return value equals the expected one
  // return
  //    [threshold>11]*[
  //            [factor>9]*66*factor        // expectedResultLhsTerm
  //          + [1-[factor>9]]*33*factor]   // expectedResultMiddleTerm
  //    + [1-[threshold>11]]*99;            // expectedResultRhsTerm

  auto expectedResultLhsTerm = new ArithmeticExpr(
      new LogicalExpr(new Variable("factor"), LogCompOp::greater, new LiteralInt(9)),
      ArithmeticOp::multiplication,
      new ArithmeticExpr(
          new LiteralInt(66),
          ArithmeticOp::multiplication,
          new Variable("factor")));
  auto expectedResultMiddleTerm = new ArithmeticExpr(
      new ArithmeticExpr(
          new LiteralInt(1),
          ArithmeticOp::subtraction,
          new LogicalExpr(new Variable("factor"), LogCompOp::greater, new LiteralInt(9))),
      ArithmeticOp::multiplication,
      new ArithmeticExpr(
          new LiteralInt(33),
          ArithmeticOp::multiplication,
          new Variable("factor")));
  auto expectedResultRhsTerm = new ArithmeticExpr(
      new ArithmeticExpr(
          new LiteralInt(1),
          ArithmeticOp::subtraction,
          new LogicalExpr(new Variable("threshold"), LogCompOp::greater, new LiteralInt(11))),
      ArithmeticOp::multiplication,
      new LiteralInt(99));

  auto expectedResult = new ArithmeticExpr(
      new ArithmeticExpr(
          new LogicalExpr(new Variable("threshold"), LogCompOp::greater, new LiteralInt(11)),
          ArithmeticOp::multiplication,
          new ArithmeticExpr(
              expectedResultLhsTerm,
              ArithmeticOp::addition,
              expectedResultMiddleTerm)),
      ArithmeticOp::addition,
      expectedResultRhsTerm);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedResult));
  // check that 'b' was memorized correctly
  EXPECT_TRUE(getVariableValue("b")->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_partiallyEvaluableOnly) { /* NOLINT */
  //  -- input --
  // int f(plaintext_int x) {
  //  int y = 42;
  //  x = x+29
  //  return x+y
  // }
  //  -- expected --
  // int f(plaintext_int x) {
  //  return x+71;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(
          new Datatype(Types::INT, false), new Variable("x"))});
  auto varDeclY = new VarDecl("y", 42);
  auto varAssignmX =
      new VarAssignm("x", new ArithmeticExpr(new Variable("x"), ArithmeticOp::addition, new LiteralInt(29)));
  auto returnStmt = new Return(
      new ArithmeticExpr(new Variable("x"), ArithmeticOp::addition, new Variable("y")));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varDeclY);
  function->addStatement(varAssignmX);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new ArithmeticExpr(
      new LiteralInt(71),
      ArithmeticOp::addition,
      new Variable("x"));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_unsupportedNestedOperator) { /* NOLINT */
  //  -- input --
  // int f(plaintext_int a) {
  //  return 9 + (34 + (22 / (a / (11 * 42))));
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return 43 + (22 / (a / 462));
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(
          new Datatype(Types::INT, false), new Variable("a"))});
  auto returnStmt = new Return(
      new ArithmeticExpr(new LiteralInt(9), addition,
                         new ArithmeticExpr(new LiteralInt(34), addition,
                                            new ArithmeticExpr(new LiteralInt(22), division,
                                                               new ArithmeticExpr(new Variable("a"), division,
                                                                                  new ArithmeticExpr(new LiteralInt(11),
                                                                                                     multiplication,
                                                                                                     42))))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new ArithmeticExpr(
      new ArithmeticExpr(new LiteralInt(22), division,
                         new ArithmeticExpr(new Variable("a"), division, new LiteralInt(462))),
      ArithmeticOp::addition,
      new LiteralInt(43));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_nestedOperatorsSimplifiableOnOneSideOnly) { /* NOLINT */
  //  -- input --
  // int f(plaintext_int a) {
  //  return (4 + (3 + 8) - (a / (2 * 4))
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return 15 - (a / 8)
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(
          new Datatype(Types::INT, false), new Variable("a"))});
  auto returnStmt = new Return(
      new ArithmeticExpr(
          new ArithmeticExpr(new LiteralInt(4), addition,
                             new ArithmeticExpr(new LiteralInt(3), addition, new LiteralInt(8))),
          subtraction,
          new ArithmeticExpr(new Variable("a"), division,
                             new ArithmeticExpr(new LiteralInt(2), multiplication, new LiteralInt(4)))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new ArithmeticExpr(new LiteralInt(15),
                                              subtraction,
                                              new ArithmeticExpr(new Variable("a"), division, new LiteralInt(8)));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_nestedLogicalOperators__EXPECTED_FAIL) { /* NOLINT */
  //  -- input --
  // int f(encrypted_bool a, plaintext_bool b, encrypted_bool c) {
  //  return (a ^ (b ^ false)) && ((true || false) || true)
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return a ^ b;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {
          new FunctionParameter(new Datatype(Types::BOOL, true), new Variable("a")),
          new FunctionParameter(new Datatype(Types::BOOL, false), new Variable("b")),
          new FunctionParameter(new Datatype(Types::BOOL, true), new Variable("c"))
      });
  auto returnStmt = new Return(
      new LogicalExpr(new LogicalExpr(new Variable("a"), logicalXor,
                                      new LogicalExpr(
                                          new Variable("b"),
                                          logicalXor,
                                          new LiteralBool(false))),
                      logicalAnd,
                      new LogicalExpr(new LogicalExpr(
                          new LiteralBool(true),
                          logicalOr,
                          new LiteralBool(false)),
                                      logicalOr,
                                      new LiteralBool(true))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new LogicalExpr(new Variable("a"), logicalXor, new Variable("b"));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));

  // check that at the end of the evaluation traversal, the removableNodes map is empty
  EXPECT_EQ(ctes.removableNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_logicalAndSimplification_ANDtrue) { /* NOLINT */
  //  -- input --
  // int f(encrypted_bool a, plaintext_bool b) {
  //  return a && (true && b);
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return a && b;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {
          new FunctionParameter(new Datatype(Types::BOOL, true), new Variable("a")),
          new FunctionParameter(new Datatype(Types::BOOL, false), new Variable("b"))
      });
  auto returnStmt = new Return(
      new LogicalExpr(
          new Variable("a"),
          logicalAnd,
          new LogicalExpr(
              new LiteralBool(true),
              logicalAnd,
              new Variable("b"))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new LogicalExpr(new Variable("a"), logicalAnd, new Variable("b"));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_logicalAndSimplification_ANDfalse) { /* NOLINT */
  //  -- input --
  // int f(encrypted_bool a, plaintext_bool b) {
  //  return a && (false && b);
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return false;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {
          new FunctionParameter(new Datatype(Types::BOOL, true), new Variable("a")),
          new FunctionParameter(new Datatype(Types::BOOL, false), new Variable("b"))
      });
  auto returnStmt = new Return(
      new LogicalExpr(
          new Variable("a"),
          logicalAnd,
          new LogicalExpr(
              new LiteralBool(false),
              logicalAnd,
              new Variable("b"))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new LiteralBool(false);
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_logicalAndSimplification_ORfalse) { /* NOLINT */
  //  -- input --
  // int f(encrypted_bool a, plaintext_bool b) {
  //  return a || (b || false);
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return a || b;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {
          new FunctionParameter(new Datatype(Types::BOOL, true), new Variable("a")),
          new FunctionParameter(new Datatype(Types::BOOL, false), new Variable("b"))
      });
  auto returnStmt = new Return(
      new LogicalExpr(
          new Variable("a"),
          logicalOr,
          new LogicalExpr(
              new Variable("b"),
              logicalOr,
              new LiteralBool(false))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new LogicalExpr(new Variable("a"), logicalOr, new Variable("b"));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_logicalAndSimplification_ORtrue) { /* NOLINT */
  //  -- input --
  // int f(encrypted_bool a, plaintext_bool b) {
  //  return a || (b || true);
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return true;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {
          new FunctionParameter(new Datatype(Types::BOOL, true), new Variable("a")),
          new FunctionParameter(new Datatype(Types::BOOL, false), new Variable("b"))
      });
  auto returnStmt = new Return(
      new LogicalExpr(
          new Variable("a"),
          logicalOr,
          new LogicalExpr(
              new Variable("b"),
              logicalOr,
              new LiteralBool(true))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new LiteralBool(true);
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_logicalAndSimplification_XORtrue) {
  //  -- input --
  // int f(encrypted_bool a, plaintext_bool b) {
  //  return a ^ (b ^ true);
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return a ^ !b;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {
          new FunctionParameter(new Datatype(Types::BOOL, true), new Variable("a")),
          new FunctionParameter(new Datatype(Types::BOOL, false), new Variable("b"))
      });
  auto returnStmt = new Return(
      new LogicalExpr(
          new Variable("a"),
          logicalXor,
          new LogicalExpr(
              new Variable("b"),
              logicalXor,
              new LiteralBool(true))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new LogicalExpr(new Variable("a"), logicalXor, new UnaryExpr(negation, new Variable("b")));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));
}

TEST_F(CompileTimeExpressionSimplifierFixture, symbolicTerms_logicalAndSimplification_XORfalse) {
  //  -- input --
  // int f(encrypted_bool a, plaintext_bool b) {
  //  return a ^ (b ^ false);
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return a ^ b;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {
          new FunctionParameter(new Datatype(Types::BOOL, true), new Variable("a")),
          new FunctionParameter(new Datatype(Types::BOOL, false), new Variable("b"))
      });
  auto returnStmt = new Return(
      new LogicalExpr(
          new Variable("a"),
          logicalXor,
          new LogicalExpr(
              new Variable("b"),
              logicalXor,
              new LiteralBool(false))));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  auto expectedReturnVal = new LogicalExpr(new Variable("a"), logicalXor, new Variable("b"));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedReturnVal));
}

TEST_F(CompileTimeExpressionSimplifierFixture, WhileLoop_compileTimeKnownExpression_removalExpected) { /* NOLINT */
  //  -- input --
  // int f(plaintext_int a) {
  //  int i = 2;
  //  while (i > 10) {
  //    a = a * a;
  //    i = i-1;
  //  }
  //  return a;
  // }
  //  -- expected --
  // int f(plaintext_int a) {
  //  return a;
  // }
  auto function = new Function("f");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(new Datatype(Types::INT, true), new Variable("i"))});
  function->setParameterList(functionParamList);

  function->addStatement(new VarDecl("i", 2));

  auto whileBody =
      new Block({
                    new VarAssignm("a",
                                   new ArithmeticExpr(
                                       new Variable("a"),
                                       multiplication,
                                       new Variable("a"))),
                    new VarAssignm("i",
                                   new ArithmeticExpr(
                                       new Variable("i"),
                                       subtraction,
                                       new LiteralInt(1)))});
  function->addStatement(new While(new LogicalExpr(
      new Variable("i"),
      greater,
      new LiteralInt(10)), whileBody));

  auto *returnStatement = new Return(new Variable("a"));
  function->addStatement(returnStatement);

  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that simplification generated the expected simplified AST
  EXPECT_EQ(function->getBodyStatements().size(), 1);
  EXPECT_TRUE(function->getBodyStatements().front()->isEqual(new Return(new Variable("a"))));
}

TEST_F(CompileTimeExpressionSimplifierFixture, Call_inliningExpected) { /* NOLINT */
  //  -- input --
  // int f() {
  //  return computeX(a);       --> int computeX(plaintext_int x) { return x + 111; }
  // }
  //  -- expected --
  // int f() {
  //  return a + 111;
  // }
  auto function = new Function("f");
  auto funcComputeX =
      new Function("computeX",
                   new ParameterList({new FunctionParameter(new Datatype(Types::INT, false), new Variable("x"))}),
                   new Block(new Return(new ArithmeticExpr(new Variable("x"), addition, new LiteralInt(111)))));
  auto callStmt = new Call(
      {new FunctionParameter(new Datatype(Types::INT, false), new Variable("a"))}, funcComputeX);

  auto returnStatement = new Return(callStmt);

  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  auto expectedReturnExpr = new ArithmeticExpr(new Variable("a"), addition, new LiteralInt(111));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedReturnExpr));
}

TEST_F(CompileTimeExpressionSimplifierFixture, Call_inliningExpected2) { /* NOLINT */
  //  -- input --
  // int f() {
  //  return computeX(232);       --> int computeX(plaintext_int x) { return x + 111; }
  // }
  //  -- expected --
  // int f() {
  //  return 343;
  // }
  auto function = new Function("f");
  auto funcComputeX =
      new Function("computeX",
                   new ParameterList({new FunctionParameter(new Datatype(Types::INT, false), new Variable("x"))}),
                   new Block(new Return(new ArithmeticExpr(new Variable("x"), addition, new LiteralInt(111)))));
  auto callStmt = new Call(
      {new FunctionParameter(
          new Datatype(Types::INT, false), new LiteralInt(232))}, funcComputeX);

  auto returnStatement = new Return(callStmt);

  function->addStatement(returnStatement);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  auto expectedReturnExpr = new LiteralInt(343);
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedReturnExpr));
}

TEST_F(CompileTimeExpressionSimplifierFixture, Rotate_executionExpected) {
  //  -- input --
  // rotateVec(int inputA) {
  //   int sumVec = {{1, 7, 3}};   // [1 7 3]
  //   return sumVec.rotate(1);    // [3 1 7]
  // }
  //  -- expected --
  // rotateVec(int inputA) {
  //   return LiteralInt{{3, 1, 7}};
  // }
  Ast ast;
  AstTestingGenerator::generateAst(24, ast);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  auto expectedReturnExpr = new LiteralInt(new Matrix<int>({{3, 1, 7}}));
  auto returnStatement = ast.getRootNode()->castTo<Function>()->getBodyStatements().back()->castTo<Return>();
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedReturnExpr));
}
