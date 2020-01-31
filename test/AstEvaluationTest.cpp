#include <Ast.h>
#include <Operator.h>
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"
#include "FunctionParameter.h"
#include "Variable.h"
#include "BinaryExpr.h"
#include "Return.h"
#include "Function.h"
#include "Call.h"
#include "VarAssignm.h"

TEST(AstEvaluationTests, simpleAstEvaluation1) {
    Ast ast;
    AstTestingGenerator::generateAst(7, ast);
    std::map<std::string, Literal *> params = {
            {"width",  new LiteralInt(31)},
            {"length", new LiteralInt(87)},
            {"depth",  new LiteralInt(771)}
    };
    auto result = dynamic_cast<LiteralInt *>(ast.evaluate(params, false));
    ASSERT_EQ(*result, *new LiteralInt(693'129));
}

TEST(AstEvaluationTests, simpleAstEvaluationIfThenBranch) {
  Ast ast;
  AstTestingGenerator::generateAst(8, ast);
    std::map<std::string, Literal *> params = {
            {"inputA", new LiteralInt(852)},
            {"inputB", new LiteralInt(7)},
            {"takeIf", new LiteralBool(true)}
    };
    auto result = dynamic_cast<LiteralInt *>(ast.evaluate(params, false));
  ASSERT_EQ(*result, *new LiteralInt(0));
}

TEST(AstEvaluationTests, simpleAstEvaluationIfElseBranch) {
  Ast ast;
  AstTestingGenerator::generateAst(8, ast);
    std::map<std::string, Literal *> params = {
            {"inputA", new LiteralInt(852)},
            {"inputB", new LiteralInt(7)},
            {"takeIf", new LiteralBool(false)}
    };
    auto result = dynamic_cast<LiteralInt *>(ast.evaluate(params, false));
  ASSERT_EQ(*result, *new LiteralInt(5943));
}

TEST(AstEvaluationTests, simpleAstEvaluationUnaryLogExpr1) {
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);
    std::map<std::string, Literal *> params = {
            {"inputA", new LiteralInt(11)},
            {"inputB", new LiteralInt(-500)},
            {"strong", new LiteralBool(true)},
            {"negate", new LiteralBool(true)}
    };
    auto result = dynamic_cast<LiteralBool *>(ast.evaluate(params, false));
  ASSERT_EQ(*result, *new LiteralBool(true));
}

TEST(AstEvaluationTests, simpleAstEvaluationUnaryLogExpr2) {
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);
    std::map<std::string, Literal *> params = {
            {"inputA", new LiteralInt(11)},
            {"inputB", new LiteralInt(12)},
            {"strong", new LiteralBool(true)},
            {"negate", new LiteralBool(true)}
    };
    auto result = dynamic_cast<LiteralBool *>(ast.evaluate(params, false));
  ASSERT_EQ(*result, *new LiteralBool(false));
}

TEST(AstEvaluationTests, simpleAstEvaluationUnaryLogExpr3) {
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);
    std::map<std::string, Literal *> params = {
            {"inputA", new LiteralInt(11)},
            {"inputB", new LiteralInt(461)},
            {"strong", new LiteralBool(true)},
            {"negate", new LiteralBool(false)}
    };
    auto result = dynamic_cast<LiteralBool *>(ast.evaluate(params, false));
  ASSERT_EQ(*result, *new LiteralBool(true));
}

TEST(AstEvaluationTests, simpleAstEvaluationStringConcat) {
  Ast ast;
  AstTestingGenerator::generateAst(10, ast);
    std::map<std::string, Literal *> params = {
            {"strA", new LiteralString("hello ")},
            {"strB", new LiteralString("world!")}
    };
    auto result = dynamic_cast<LiteralString *>(ast.evaluate(params, false));
  ASSERT_EQ(*result, *new LiteralString("hello world!"));
}

TEST(AstEvaluationTests, complexAstEvaluationWhileNotExecuted) {
  Ast ast;
  AstTestingGenerator::generateAst(11, ast);
    std::map<std::string, Literal *> params = {
            {"encryptedA", new LiteralInt(1)},
            {"encryptedB", new LiteralInt(7)},
            {"randInt",    new LiteralInt(4)}
    };
    auto result = dynamic_cast<LiteralInt *>(ast.evaluate(params, false));
  ASSERT_EQ(*result, *new LiteralInt(0));
}

TEST(AstEvaluationTests, complexAstEvaluationWhileExecutedThreeTimes) {
  Ast ast;
  AstTestingGenerator::generateAst(11, ast);
    std::map<std::string, Literal *> params = {
            {"encryptedA", new LiteralInt(1'182)},
            {"encryptedB", new LiteralInt(7)},
            {"randInt",    new LiteralInt(3)}
    };
    auto result = dynamic_cast<LiteralInt *>(ast.evaluate(params, false));
  ASSERT_EQ(*result, *new LiteralInt(21));
}

TEST(AstEvaluationTests, complexAstEvaluationWithNestedFunctionCall_LiteralParameterValue) {
    Ast ast;
    AstTestingGenerator::generateAst(12, ast);
    std::map<std::string, Literal *> params;
    auto result = dynamic_cast<LiteralInt *>(ast.evaluate(params, false));
    ASSERT_EQ(*result, *new LiteralInt(1'056));
}

TEST(AstEvaluationTests, complexAstEvaluationWithNestedFunctionCall_BexpParameterValue) {
    Ast ast;
    AstTestingGenerator::generateAst(13, ast);
    std::map<std::string, Literal *> params;
    auto result = dynamic_cast<LiteralInt *>(ast.evaluate(params, false));
    ASSERT_EQ(*result, *new LiteralInt(7'168));
}
