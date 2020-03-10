#include "Ast.h"
#include "Operator.h"
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"
#include "FunctionParameter.h"
#include "Variable.h"
#include "ArithmeticExpr.h"
#include "Return.h"
#include "Function.h"
#include "Call.h"
#include "VarAssignm.h"

TEST(AstEvaluationTests, simpleAstEvaluation1) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(7, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"width", new LiteralInt(31)},
      {"length", new LiteralInt(87)},
      {"depth", new LiteralInt(771)}
  };
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralInt(693'129));
}

TEST(AstEvaluationTests, simpleAstEvaluationIfThenBranch) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(8, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"inputA", new LiteralInt(852)},
      {"inputB", new LiteralInt(7)},
      {"takeIf", new LiteralBool(true)}
  };
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralInt(0));
}

TEST(AstEvaluationTests, simpleAstEvaluationIfElseBranch) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(8, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"inputA", new LiteralInt(852)},
      {"inputB", new LiteralInt(7)},
      {"takeIf", new LiteralBool(false)}
  };
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralInt(5943));
}

TEST(AstEvaluationTests, simpleAstEvaluationUnaryLogExpr1) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"inputA", new LiteralInt(11)},
      {"inputB", new LiteralInt(-500)},
      {"strong", new LiteralBool(true)},
      {"negate", new LiteralBool(true)}
  };
  auto result = dynamic_cast<LiteralBool *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralBool(true));
}

TEST(AstEvaluationTests, simpleAstEvaluationUnaryLogExpr2) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"inputA", new LiteralInt(11)},
      {"inputB", new LiteralInt(12)},
      {"strong", new LiteralBool(true)},
      {"negate", new LiteralBool(true)}
  };
  auto result = dynamic_cast<LiteralBool *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralBool(false));
}

TEST(AstEvaluationTests, simpleAstEvaluationUnaryLogExpr3) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"inputA", new LiteralInt(11)},
      {"inputB", new LiteralInt(461)},
      {"strong", new LiteralBool(true)},
      {"negate", new LiteralBool(false)}
  };
  auto result = dynamic_cast<LiteralBool *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralBool(true));
}

TEST(AstEvaluationTests, simpleAstEvaluationStringConcat) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(10, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"strA", new LiteralString("hello ")},
      {"strB", new LiteralString("world!")}
  };
  auto result = dynamic_cast<LiteralString *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralString("hello world!"));
}

TEST(AstEvaluationTests, complexAstEvaluationWhileNotExecuted) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(11, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"encryptedA", new LiteralInt(1)},
      {"encryptedB", new LiteralInt(7)},
      {"randInt", new LiteralInt(4)}
  };
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralInt(0));
}

TEST(AstEvaluationTests, complexAstEvaluationWhileExecutedThreeTimes) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(11, ast);
  std::unordered_map<std::string, AbstractLiteral *> params = {
      {"encryptedA", new LiteralInt(1'182)},
      {"encryptedB", new LiteralInt(7)},
      {"randInt", new LiteralInt(3)}
  };
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralInt(21));
}

TEST(AstEvaluationTests, complexAstEvaluationWithNestedFunctionCall_LiteralParameterValue) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(12, ast);
  std::unordered_map<std::string, AbstractLiteral *> params;
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralInt(1'056));
}

TEST(AstEvaluationTests, complexAstEvaluationWithNestedFunctionCall_AexpParameterValue) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(13, ast);
  std::unordered_map<std::string, AbstractLiteral *> params;
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralInt(7'168));
}

TEST(AstEvaluationTests, astIncludingForLoop) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(23, ast);
  std::unordered_map<std::string, AbstractLiteral *> params({{"inputA", new LiteralInt(5)}});
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst(params, false).front());
  ASSERT_EQ(*result, *new LiteralInt(30));
}

TEST(AstEvaluationTests, astLiteralRotation) {
  Ast ast;
  AstTestingGenerator::generateAst(24, ast);
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst({}, false).front());
  auto m = *dynamic_cast<Matrix<int> *>(result->getMatrix());
  EXPECT_EQ(m(0, 0), 3);
  EXPECT_EQ(m(0, 1), 1);
  EXPECT_EQ(m(0, 2), 7);
}

TEST(AstEvaluationTests, astRotateAndSumVector) {
  Ast ast;
  AstTestingGenerator::generateAst(25, ast);
  auto result = dynamic_cast<LiteralInt *>(ast.evaluateAst({}, false).front());
  auto m = *dynamic_cast<Matrix<int> *>(result->getMatrix());
  EXPECT_EQ(m.values, std::vector<std::vector<int>>({{11, 11, 11}}));
}
