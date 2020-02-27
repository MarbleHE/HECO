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
      OpSymb::multiplication,
      new LiteralInt(11));
  auto varAssignm = new VarDecl("alpha", new Datatype(Types::INT, false), arithmeticExpr);

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
  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
      OpSymb::multiplication,
      new ArithmeticExpr(new LiteralInt(4),
                         OpSymb::multiplication,
                         new LiteralInt(7)));
  auto varAssignm = new VarDecl("alpha", new Datatype(Types::INT, false), arithmeticExpr);

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varAssignm);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is no computed variable
  EXPECT_TRUE(ctes.variableValues.empty());
  // check that the rhs operand of arithmeticExpr is simplified
  EXPECT_EQ(arithmeticExpr->getRight()->castTo<LiteralInt>()->getValue(), 28);

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
      OpSymb::multiplication,
      new ArithmeticExpr(new LiteralInt(4),
                         OpSymb::multiplication,
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
      OpSymb::multiplication,
      new ArithmeticExpr(new LiteralInt(4),
                         OpSymb::multiplication,
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
  EXPECT_THROW(getVariableValue("alpha"), std::logic_error);

  // check that none of the nodes are deleted
  EXPECT_EQ(numberOfNodesBeforeSimplification, ast.getAllNodes().size());

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, logicalExpr_literalsOnly_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //  plaintext_bool alpha = true && false;
  // }
  auto function = new Function("compute");
  auto logicalExpr = new LogicalExpr(
      new LiteralBool(true),
      OpSymb::logicalAnd,
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, logicalExpr_variableUnknown_lhsOperandEvaluableOnly) { /* NOLINT */
  // void compute(encrypted_bool encryptedA) {
  //  plaintext_bool alpha = (true ^ false) || encryptedA;
  // }
  auto function = new Function("compute");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(new Datatype(Types::BOOL, true),
                             new Variable("encryptedA"))});
  auto logicalExpr = new LogicalExpr(
      new LogicalExpr(new LiteralBool(true),
                      OpSymb::logicalXor,
                      new LiteralBool(false)),
      OpSymb::logicalOr,
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

  // check that there is no computed variable
  EXPECT_TRUE(ctes.variableValues.empty());
  // check that the lhs operand of arithmeticExpr is simplified
  EXPECT_EQ(logicalExpr->getLeft()->castTo<LiteralBool>()->getValue(), true);

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
      OpSymb::logicalOr,
      new LogicalExpr(new LiteralBool(false),
                      OpSymb::logicalAnd,
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, logicalExpr_variablesUnknown_notAnythingEvaluable) { /* NOLINT */
  // void compute(encrypted_bool encryptedA, plaintext_bool paramB) {
  //  plaintext_bool alpha = encryptedA && (true ^ encryptedB);
  // }
  auto function = new Function("compute");
  auto functionParameters = std::vector<FunctionParameter *>(
      {new FunctionParameter(new Datatype(Types::BOOL, true),
                             new Variable("encryptedA")),
       new FunctionParameter(new Datatype(Types::BOOL, false),
                             new Variable("paramB"))});
  function->setParameterList(new ParameterList(functionParameters));

  auto logicalExpr = new LogicalExpr(
      new Variable("encryptedA"),
      OpSymb::logicalAnd,
      new LogicalExpr(new LiteralBool(true),
                      OpSymb::logicalXor,
                      new Variable("encryptedB")));
  auto varDeclAlpha = new VarDecl("alpha",
                                  new Datatype(Types::BOOL, false),
                                  logicalExpr);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);
  auto numberOfNodesBeforeSimplification = ast.getAllNodes().size();

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
  EXPECT_THROW(getVariableValue("alpha"), std::logic_error);

  // check that none of the nodes are deleted
  EXPECT_EQ(numberOfNodesBeforeSimplification, ast.getAllNodes().size());

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, unaryExpr_literalsOnly_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //  plaintext_bool truthValue = !false;
  // }
  auto function = new Function("compute");
  auto unaryExpr = new UnaryExpr(OpSymb::negation, new LiteralBool(false));
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, unaryExpr_variableKnown_fullyEvaluable) { /* NOLINT */
  // void compute() {
  //  plaintext_bool alpha = true;
  //  plaintext_bool beta = !alpha;
  // }
  auto function = new Function("compute");
  auto varDeclParameterA = new VarDecl("alpha", true);
  auto unaryExpr = new UnaryExpr(OpSymb::negation, new Variable("alpha"));
  auto varDeclAlpha = new VarDecl("beta",
                                  new Datatype(Types::BOOL, false),
                                  unaryExpr);

  // connect objects
  function->addStatement(varDeclParameterA);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that 'alpha' is computed correctly
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
  auto unaryExpr = new UnaryExpr(OpSymb::negation, new Variable("paramA"));
  auto varDeclAlpha = new VarDecl("beta",
                                  new Datatype(Types::BOOL, false),
                                  unaryExpr);

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varDeclAlpha);
  ast.setRootNode(function);
  auto numberOfNodesBeforeSimplification = ast.getAllNodes().size();

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // check that there is no computed variable
  EXPECT_TRUE(ctes.variableValues.empty());
  // check that none of the nodes are deleted
  EXPECT_EQ(numberOfNodesBeforeSimplification, ast.getAllNodes().size());

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, varAssignm_variableDeclarationOnly) { /* NOLINT */
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

  // check that 'alpha' is computed correctly
  EXPECT_EQ(getVariableValue("alpha")->castTo<LiteralFloat>()->getValue(), 0.0f);

  // check that the statement VarDecl and its children are deleted
  EXPECT_EQ(function->getBody()->getStatements().size(), 0);

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, varAssignm_symbolicTerms_circularDependency) { /* NOLINT */
  // int Foo(plaintext_int x, plaintext_int y) {
  //  x = y+3
  //  y = x+2
  //  return x+y
  // }
  auto function = new Function("Foo");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(
          new Datatype(Types::INT, false), new Variable("x")),
       new FunctionParameter(
           new Datatype(Types::INT, false), new Variable("y"))});
  auto varAssignmX = new VarAssignm("x",
                                    new ArithmeticExpr(new Variable("y"), OpSymb::addition, 3));
  auto varAssignmY = new VarAssignm("y",
                                    new ArithmeticExpr(new Variable("x"), OpSymb::addition, 2));
  auto returnStmt = new Return(
      new ArithmeticExpr(new Variable("x"), OpSymb::addition, new Variable("y")));

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

  DotPrinter().printAsDotFormattedGraph(ast);

  // check that simplification generated the expected simplified AST
  auto expectedAst = new ArithmeticExpr(
      new ArithmeticExpr(
          new Variable("y"),
          OpSymb::addition,
          new LiteralInt(3)),
      OpSymb::addition,
      new ArithmeticExpr(
          new ArithmeticExpr(new Variable("y"), OpSymb::addition, new LiteralInt(3)),
          OpSymb::addition,
          new LiteralInt(2)));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedAst));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}

TEST_F(CompileTimeExpressionSimplifierFixture, return_literalOnly_expectedNoChange) { /* NOLINT */
  // float compute() {
  //  return 42.24;
  // }
  // -- expected –-
  // no change
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
                         OpSymb::addition,
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
                         OpSymb::addition,
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
  auto functionParam = new ParameterList({new FunctionParameter(new Datatype(Types::INT), new Variable("a"))});
  auto varDecl = new VarDecl("b",
                             new Datatype(Types::INT),
                             new ArithmeticExpr(
                                 new LiteralInt(3),
                                 OpSymb::addition,
                                 new LiteralInt(4)));
  auto returnStatement =
      new Return({
                     new ArithmeticExpr(
                         new Variable("a"),
                         OpSymb::multiplication,
                         new Variable("b")),
                     new ArithmeticExpr(
                         new LiteralInt(2),
                         OpSymb::subtraction,
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
                  ->isEqual(new ArithmeticExpr(new Variable("a"), OpSymb::multiplication, new LiteralInt(7))));
  // check return expression 2: -5
  EXPECT_TRUE(returnStatement->getReturnExpressions().at(1)->isEqual(new LiteralInt(-5)));
  // check return expression 3: 21
  EXPECT_TRUE(returnStatement->getReturnExpressions().at(2)->isEqual(new LiteralInt(21)));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
              OpSymb::addition,
              new LiteralInt(12)),
          OpSymb::greater,
          new LiteralInt(20)),
      new VarAssignm("a", new ArithmeticExpr(
          new Variable("a"),
          OpSymb::multiplication,
          new LiteralInt(2))));
  auto returnStatement =
      new Return(new ArithmeticExpr(new Variable("a"),
                                    OpSymb::multiplication,
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
      new ArithmeticExpr(new Variable("b"), OpSymb::addition, new LiteralInt(12)),
      OpSymb::greater,
      new LiteralInt(20));
  auto thenBranch = new VarAssignm("a", new ArithmeticExpr(
      new Variable("a"),
      OpSymb::multiplication,
      new LiteralInt(2)));
  auto elseBranch = new VarAssignm("a", new LiteralInt(1));
  auto ifStmt = new If(condition, thenBranch, elseBranch);

  auto returnStatement =
      new Return(new ArithmeticExpr(new Variable("a"),
                                    OpSymb::multiplication,
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

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
      new ArithmeticExpr(new Variable("b"), OpSymb::addition, new LiteralInt(12)),
      OpSymb::smaller,
      new LiteralInt(20));
  auto thenBranch = new VarAssignm("a", new ArithmeticExpr(
      new Variable("a"),
      OpSymb::multiplication,
      new LiteralInt(2)));
  auto elseBranch = new VarAssignm("a", new LiteralInt(1));
  auto ifStmt = new If(condition, thenBranch, elseBranch);

  auto returnStatement =
      new Return(new ArithmeticExpr(new Variable("a"),
                                    OpSymb::multiplication,
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
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(new LiteralInt(32)));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
                                         OpSymb::greater,
                                         new LiteralInt(20));
  auto ifStmt = new If(
      ifStmtCondition,
      new VarAssignm("b", new ArithmeticExpr(
          new LiteralInt(2),
          OpSymb::multiplication,
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
                          OpSymb::greater,
                          new LiteralInt(20)),
          OpSymb::multiplication,
          new LiteralInt(44)),
      OpSymb::addition,
      new ArithmeticExpr(
          new ArithmeticExpr(
              new LiteralInt(1),
              OpSymb::subtraction,
              new LogicalExpr(new Variable("a"),
                              OpSymb::greater,
                              new LiteralInt(20))),
          OpSymb::multiplication,
          new LiteralInt(22)));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
                                         OpSymb::greater,
                                         new LiteralInt(20));
  auto thenStatements = std::vector<AbstractStatement *>(
      {new VarDecl("c", 642),
       new VarAssignm("b",
                      new ArithmeticExpr(
                          new ArithmeticExpr(new LiteralInt(2), OpSymb::multiplication, new Variable("c")),
                          OpSymb::subtraction,
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
                          OpSymb::greater,
                          new LiteralInt(20)),
          OpSymb::multiplication,
          new LiteralInt(1'283));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
                                         OpSymb::greater,
                                         new LiteralInt(20));
  auto thenStatements = std::vector<AbstractStatement *>(
      {new VarDecl("c", 642),
       new VarAssignm("b",
                      new ArithmeticExpr(
                          new ArithmeticExpr(new LiteralInt(2), OpSymb::multiplication, new Variable("c")),
                          OpSymb::subtraction,
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
                          OpSymb::greater,
                          new LiteralInt(20)),
          OpSymb::multiplication,
          new LiteralInt(1'283)),
      OpSymb::addition,
      new ArithmeticExpr(new ArithmeticExpr(
          new LiteralInt(1),
          OpSymb::subtraction,
          new LogicalExpr(new Variable("a"),
                          OpSymb::greater,
                          new LiteralInt(20))),
                         OpSymb::multiplication,
                         new LiteralInt(42)));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
                                         OpSymb::smaller,
                                         new LiteralInt(11));
  auto thenBranch = new VarAssignm("b",
                                   new ArithmeticExpr(
                                       new LiteralInt(2),
                                       OpSymb::multiplication,
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
                          OpSymb::smaller,
                          new LiteralInt(11)),
          OpSymb::multiplication,
          new ArithmeticExpr(new LiteralInt(2), OpSymb::multiplication, new Variable("factor"))),
      OpSymb::addition,
      new ArithmeticExpr(new ArithmeticExpr(
          new LiteralInt(1),
          OpSymb::subtraction,
          new LogicalExpr(new Variable("threshold"),
                          OpSymb::smaller,
                          new LiteralInt(11))),
                         OpSymb::multiplication,
                         new Variable("factor")));
  EXPECT_TRUE(returnStatement->getReturnExpressions().front()->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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

  auto innerIfStatementCondition = new LogicalExpr(new Variable("factor"), OpSymb::greater, new LiteralInt(9));
  auto innerIfStatement = new If(innerIfStatementCondition,
                                 new VarAssignm("b", new ArithmeticExpr(
                                     new ArithmeticExpr(new Variable("b"), OpSymb::multiplication, 2),
                                     OpSymb::multiplication,
                                     new Variable("factor"))),
                                 new VarAssignm("b",
                                                new ArithmeticExpr(new Variable("b"),
                                                                   OpSymb::multiplication,
                                                                   new Variable("factor"))));

  auto outerIfStmtThenBlock =
      new Block({new VarAssignm("b", new ArithmeticExpr(new Variable("b"), OpSymb::division, new LiteralInt(3))),
                 innerIfStatement});
  auto outerIfStatementCondition = new LogicalExpr(new Variable("threshold"), OpSymb::greater, new LiteralInt(11));
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
      new LogicalExpr(new Variable("factor"), OpSymb::greater, new LiteralInt(9)),
      OpSymb::multiplication,
      new ArithmeticExpr(
          new LiteralInt(66),
          OpSymb::multiplication,
          new Variable("factor")));
  auto expectedResultMiddleTerm = new ArithmeticExpr(
      new ArithmeticExpr(
          new LiteralInt(1),
          OpSymb::subtraction,
          new LogicalExpr(new Variable("factor"), OpSymb::greater, new LiteralInt(9))),
      OpSymb::multiplication,
      new ArithmeticExpr(
          new LiteralInt(33),
          OpSymb::multiplication,
          new Variable("factor")));
  auto expectedResultRhsTerm = new ArithmeticExpr(
      new ArithmeticExpr(
          new LiteralInt(1),
          OpSymb::subtraction,
          new LogicalExpr(new Variable("threshold"), OpSymb::greater, new LiteralInt(11))),
      OpSymb::multiplication,
      new LiteralInt(99));

  auto expectedResult = new ArithmeticExpr(
      new ArithmeticExpr(
          new LogicalExpr(new Variable("threshold"), OpSymb::greater, new LiteralInt(11)),
          OpSymb::multiplication,
          new ArithmeticExpr(
              expectedResultLhsTerm,
              OpSymb::addition,
              expectedResultMiddleTerm)),
      OpSymb::addition,
      expectedResultRhsTerm);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedResult));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
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
  auto function = new Function("compute");
  auto functionParamList = new ParameterList(
      {new FunctionParameter(
          new Datatype(Types::INT, false), new Variable("x"))});
  auto varDeclY = new VarDecl("y", 42);
  auto varAssignmX = new VarAssignm("x", new ArithmeticExpr(new Variable("x"), OpSymb::addition, new LiteralInt(29)));
  auto returnStmt = new Return(
      new ArithmeticExpr(new Variable("x"), OpSymb::addition, new Variable("y")));

  // connect objects
  function->setParameterList(functionParamList);
  function->addStatement(varDeclY);
  function->addStatement(varAssignmX);
  function->addStatement(returnStmt);
  ast.setRootNode(function);

  // perform the compile-time expression simplification
  ctes.visit(ast);

  // TODO implement handling of evalu

  DotPrinter().printAsDotFormattedGraph(ast);

  // check that simplification generated the expected simplified AST
  auto expectedAst = new ArithmeticExpr(
      new ArithmeticExpr(
          new Variable("y"),
          OpSymb::addition,
          new LiteralInt(3)),
      OpSymb::addition,
      new ArithmeticExpr(
          new ArithmeticExpr(new Variable("y"), OpSymb::addition, new LiteralInt(3)),
          OpSymb::addition,
          new LiteralInt(2)));
  EXPECT_EQ(returnStmt->getReturnExpressions().size(), 1);
  EXPECT_TRUE(returnStmt->getReturnExpressions().front()->isEqual(expectedAst));

  // check that at the end of the evaluation traversal, the evalutedNodes map is empty
  EXPECT_EQ(ctes.evaluatedNodes.size(), 0);
}





// TODO(pjattke): implement tests for symbolic terms

// TODO(pjattke): write test for If statement like ifStmt_conditionValueIsUnknown_thenBranchOnlyExists_expectedRewriting
//   - with additional else branch where variables in then and else branch only partially overlap

// TODO(pjattke): write tests for Call including Function, FunctionParameter, and Block
//  - Call with Function that is expected to be replaced
//  - implementation details
//    - inline Call only if it can be evaluated at compile-time and returns a Literal (e.g., sqrt(5))
//    - add a few ASTs to test this!

// TODO(pjattke): write tests for While statement
//  - implementation details
//    - if condition is always False -> delete the loop
//    - if condition is always True -> throw an error
//    - if condition contains encrypted variables -> throw an error
//    - otherwise: unroll loop
//  - While with unknown loop condition -> cannot be evaluated
//  - While with known loop condition -> can be evaluated
//  - While that has a known loop condition but contains a non-evaluable statements

// TODO(pjattke): write a Statistics visitors that collects:
//  - total number of AND/XOR gates
//    - number of multiplications (i.e., AND gates) per level
//  - multiplicative depth (use existing MultiplicativeDepthCalculator)

// TODO(pjattke): Think about an example where current optimizer returns a worse AST than the original one, considering
//  the constraints of FHE (e.g., multiplicative depth)

// TODO(pjattke): Play with SEAL as preparatory step for next tasks - work through tutorial examples!
