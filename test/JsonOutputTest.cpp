#include <fstream>
#include "BinaryExpr.h"
#include "Block.h"
#include "Call.h"
#include "CallExternal.h"
#include "Function.h"
#include "FunctionParameter.h"
#include "Group.h"
#include "If.h"
#include "LiteralBool.h"
#include "LiteralInt.h"
#include "LiteralFloat.h"
#include "LiteralString.h"
#include "LogicalExpr.h"
#include "Return.h"
#include "UnaryExpr.h"
#include "VarAssignm.h"
#include "VarDecl.h"
#include "Variable.h"
#include "While.h"

#include "gtest/gtest.h"

/// This test suite verifies the correctness of the JSON outputs.
/// For comparison, the JSON expected objects are manually created (for simple objects) or parsed from the corresponding
/// JSON file to be found in test/aux/JsonOutputTest (for more complex objects).


TEST(JsonOutputTest, LiteralInt) { /* NOLINT */
  int val = 2;
  auto* lint = new LiteralInt(val);
  json j = {{"type", "LiteralInt"},
            {"value", val}};
  EXPECT_EQ(lint->toJson(), j);
}

TEST(JsonOutputTest, LiteralFloat) { /* NOLINT */
  float val = 33.214;
  auto* lint = new LiteralFloat(val);
  json j = {{"type", "LiteralFloat"},
            {"value", val}};
  EXPECT_EQ(lint->toJson(), j);
}

TEST(JsonOutputTest, LiteralBool) { /* NOLINT */
  bool val = true;
  auto* lbool = new LiteralBool(val);
  json j = {{"type", "LiteralBool"},
            {"value", val}};
  EXPECT_EQ(lbool->toJson(), j);
}

TEST(JsonOutputTest, LiteralString) { /* NOLINT */
  std::string val = "hello world!";
  auto* lString = new LiteralString(val);
  json j = {{"type", "LiteralString"},
            {"value", val}};
  EXPECT_EQ(lString->toJson(), j);
}

TEST(JsonOutputTest, VarAssignm) { /* NOLINT */
  std::string identifier = "myCustomVar";
  int val = 2;
  auto* lint = new LiteralInt(val);
  // myCustomVar = 2;
  auto* assignm = new VarAssignm(identifier, lint);
  json j = {{"type", "VarAssignm"},
            {"identifier", identifier},
            {"value", {
                {"type", "LiteralInt"},
                {"value", val}
            }
            }};
  EXPECT_EQ(assignm->toJson(), j);
}

TEST(JsonOutputTest, Variable) { /* NOLINT */
  std::string varIdentifier = "myVar";
  auto var = new Variable(varIdentifier);
  json j = {{"type", "Variable"},
            {"identifier", varIdentifier}};
  EXPECT_EQ(var->toJson(), j);
}

TEST(JsonOutputTest, VarDecl) { /* NOLINT */
  auto identifier = "numIterations";
  auto datatype = "int";
  int initializer = 3;
  // int numIterations = 3;
  auto* var = new VarDecl(identifier, initializer);
  json j = {{"type", "VarDecl"},
            {"identifier", identifier},
            {"datatype", datatype},
            {"initializer", {
                {"type", "LiteralInt"},
                {"value", initializer}}
            }
  };
  EXPECT_EQ(var->toJson(), j);
}

TEST(JsonOutputTest, BinaryExpr) { /* NOLINT */
  auto lintValue = 22;
  auto varIdentifier = "x";
  // x + 22;
  auto* bexp = new BinaryExpr(
      new Variable(varIdentifier),
      OpSymb::BinaryOp::addition,
      new LiteralInt(lintValue));
  json j = {{"type", "BinaryExpr"},
            {"leftOperand", {
                {"type", "Variable"},
                {"identifier", varIdentifier}}},
            {"operator", OpSymb::getTextRepr(OpSymb::addition)},
            {"rightOperand", {
                {"type", "LiteralInt"},
                {"value", lintValue}
            }}};
  EXPECT_EQ(bexp->toJson(), j);
}

TEST(JsonOutputTest, Return) { /* NOLINT */
  // Return x > 22;
  auto ret = new Return(
      new LogicalExpr(new Variable("x"), OpSymb::greater, 22));
  json j = {
      {"type", "Return"},
      {"value", {
          {"type", "LogicalExpr"},
          {"leftOperand", {
              {"type", "Variable"},
              {"identifier", "x"}
          }},
          {"rightOperand", {
              {"type", "LiteralInt"},
              {"value", 22}
          }},
          {"operator", ">"}}
      }};
  EXPECT_EQ(ret->toJson(), j);
}

TEST(JsonOutputTest, UnaryExpr) { /* NOLINT */
  auto varIdentifier = "x";
  // !x
  auto unaryExp = new UnaryExpr(OpSymb::UnaryOp::negation, new Variable(varIdentifier));
  json j = {{"type", "UnaryExpr"},
            {"operator", OpSymb::getTextRepr(OpSymb::negation)},
            {"rightOperand", {
                {"type", "Variable"},
                {"identifier", varIdentifier}}
            }};
  EXPECT_EQ(unaryExp->toJson(), j);
}

TEST(JsonOutputTest, FunctionParameter) { /* NOLINT */
  auto fp = new FunctionParameter("int", new Variable("y"));
  json j = {{"type", "FunctionParameter"},
            {"datatype", "int"},
            {"value", {
                {"type", "Variable"},
                {"identifier", "y"}
            }}};
  EXPECT_EQ(fp->toJson(), j);
}

TEST(JsonOutputTest, Group) { /* NOLINT */
  auto lintValue = 9883;
  auto varIdentifier = "totalItems";
  // (totalItems / 9883)
  auto gp = new Group(new BinaryExpr(
      new Variable(varIdentifier),
      OpSymb::BinaryOp::division,
      new LiteralInt(lintValue)));

  json j = {{"type", "Group"},
            {"expr", {{"type", "BinaryExpr"},
                      {"leftOperand", {
                          {"type", "Variable"},
                          {"identifier", varIdentifier}}},
                      {"operator", OpSymb::getTextRepr(OpSymb::division)},
                      {"rightOperand", {
                          {"type", "LiteralInt"},
                          {"value", lintValue}
                      }}}}};
  EXPECT_EQ(gp->toJson(), j);
}

TEST(JsonOutputTest, LogicalExpr) { /* NOLINT */
  auto varIdentifier = "numIterations";
  auto varIdentifierMax = "maxIterations";
  // numIterations < maxIterations
  auto* lexp = new LogicalExpr(
      new Variable(varIdentifier),
      OpSymb::smaller,
      new Variable(varIdentifierMax));
  json j = {{"type", "LogicalExpr"},
            {"leftOperand", {
                {"type", "Variable"},
                {"identifier", varIdentifier}}},
            {"operator", OpSymb::getTextRepr(OpSymb::smaller)},
            {"rightOperand", {
                {"type", "Variable"},
                {"identifier", varIdentifierMax}
            }}};
  EXPECT_EQ(lexp->toJson(), j);
}

TEST(JsonOutputTest, Operator) { /* NOLINT */
  auto opSub = new Operator(OpSymb::BinaryOp::subtraction);
  EXPECT_EQ(opSub->getOperatorString(), OpSymb::getTextRepr(OpSymb::subtraction));

  auto opDec = new Operator(OpSymb::UnaryOp::decrement);
  EXPECT_EQ(opDec->getOperatorString(), OpSymb::getTextRepr(OpSymb::decrement));

  auto opAnd = new Operator(OpSymb::LogCompOp::logicalAnd);
  EXPECT_EQ(opAnd->getOperatorString(), OpSymb::getTextRepr(OpSymb::logicalAnd));
}

TEST(JsonOutputTest, Block) { /* NOLINT */
  auto bl = new Block(new VarDecl("width", TYPES::INT, new LiteralInt(22)));
  json j = {{"type", "Block"},
            {"statements", {{
                                {"type", "VarDecl"},
                                {"datatype", "int"},
                                {"identifier", "width"},
                                {"initializer", {
                                    {"type", "LiteralInt"},
                                    {"value", 22}}}}}}};
  EXPECT_EQ(bl->toJson(), j);
}

TEST(JsonOutputTest, CallExternal) { /* NOLINT */
  std::vector<FunctionParameter*> funcParams = {
      new FunctionParameter("string", new LiteralString("Hello world!"))};

  // printf("Hello world!");
  auto callExt = new CallExternal("printf", funcParams);

  // read expected output from hard-coded file
  std::ifstream f("../../test/auxoutput/JsonOutputTest/CallExternal.json");
  json j = json::parse(f);

  EXPECT_EQ(callExt->AbstractStatement::toString(), j.dump());
  EXPECT_EQ(callExt->AbstractExpr::toString(), j.dump());
}

TEST(JsonOutputTest, Call) {/* NOLINT */
  // computeSecret(33)  --call to-->  int computeSecret(int inputA) { return inputA * 32; }
  auto call = new Call(
      {new FunctionParameter("int", new LiteralInt(33))},
      new Function("computeSecret",
                   {new FunctionParameter("int", new Variable("inputA"))},
                   {new Return(
                       new BinaryExpr(
                           new Variable("inputA"),
                           OpSymb::multiplication,
                           new LiteralInt(32)))
                   }));

  // retrieve expected result
  std::ifstream f("../../test/auxoutput/JsonOutputTest/Call.json");
  json expected = json::parse(f);

  EXPECT_EQ(call->AbstractStatement::toString(), expected.dump());
  EXPECT_EQ(call->AbstractExpr::toString(), expected.dump());
}

TEST(JsonOutputTest, Function) { /* NOLINT */
  // int computeX(int a, bool b, int z) {
  //    return [(a + 221) * b] + z
  // }
  auto func = new Function("computeX", {
      new FunctionParameter("int", new Variable("a")),
      new FunctionParameter("bool", new Variable("b")),
      new FunctionParameter("int", new Variable("z"))
  }, {
                               new Return(
                                   new BinaryExpr(
                                       new BinaryExpr(
                                           new Group(
                                               new BinaryExpr(
                                                   new Variable("a"),
                                                   OpSymb::addition,
                                                   new LiteralInt(221))),
                                           OpSymb::multiplication,
                                           new Variable("b")),
                                       OpSymb::addition,
                                       new Variable("z")))});

  // retrieve expected result
  std::ifstream f("../../test/auxoutput/JsonOutputTest/Function.json");
  json expected = json::parse(f);

  EXPECT_EQ(func->toJson(), expected);
}

TEST(JsonOutputTest, IfThenOnly) { /* NOLINT */
  // if (b == true && z >= 17) {
  //    isValid = true;
  // }
  auto ifStmt = new If(
      new LogicalExpr(
          new LogicalExpr(
              new Variable("b"),
              OpSymb::equal,
              new LiteralBool(true)),
          OpSymb::logicalAnd,
          new LogicalExpr(
              new Variable("z"),
              OpSymb::greaterEqual,
              new LiteralInt(17))),
      new VarAssignm("isValid", new LiteralBool(true)));

  // retrieve expected result
  std::ifstream f("../../test/auxoutput/JsonOutputTest/IfThenOnly.json");
  json expected = json::parse(f);

  EXPECT_EQ(ifStmt->toJson(), expected);
}

TEST(JsonOutputTest, If) { /* NOLINT */
  // if (b == true && z >= 17) {
  //    isValid = true;
  // } else {
  //    isValid = false;
  //    c = a-z;
  // }

  auto condition = new LogicalExpr(
      new LogicalExpr(
          new Variable("b"),
          OpSymb::equal,
          new LiteralBool(true)),
      OpSymb::logicalAnd,
      new LogicalExpr(
          new Variable("z"),
          OpSymb::greaterEqual,
          new LiteralInt(17)));

  auto thenBranch = new VarAssignm("isValid", new LiteralBool(true));

  std::vector<AbstractStatement*> elseStatement = {
      new VarAssignm("isValid", new LiteralBool(false)),
      new VarAssignm("c", new BinaryExpr(new Variable("a"), OpSymb::subtraction, new Variable("z")))
  };
  auto* elseBranch = new Block(&elseStatement);
  auto ifStmt = new If(condition, thenBranch, elseBranch);

  // retrieve expected result
  std::ifstream f("../../test/auxoutput/JsonOutputTest/If.json");
  json expected = json::parse(f);

  EXPECT_EQ(ifStmt->toJson(), expected);
}

TEST(JsonOutputTest, While) { /* NOLINT */
  // while (i < 10) {
  //   z = z*i;
  //   i++;
  // }
  std::vector<AbstractStatement*> blockStatements;
  blockStatements.emplace_back(
      new VarAssignm("z",
                     new BinaryExpr(
                         new Variable("z"),
                         OpSymb::multiplication,
                         new Variable("i"))));
  blockStatements.emplace_back(
      new VarAssignm("i", new UnaryExpr(OpSymb::increment, new Variable("i"))));

  auto whileStmt = new While(
      new LogicalExpr(new Variable("i"), OpSymb::smaller, new LiteralInt(10)),
      new Block(&blockStatements));

  // retrieve expected result
  std::ifstream f("../../test/auxoutput/JsonOutputTest/While.json");
  json expected = json::parse(f);

  EXPECT_EQ(whileStmt->toJson(), expected);
}
