#include <fstream>
#include "heco/ast/Literal.h"
#include "heco/ast/Variable.h"
#include "heco/ast/Assignment.h"
#include "heco/ast/VariableDeclaration.h"
#include "gtest/gtest.h"

/// This test suite verifies the correctness of the JSON outputs.
/// For comparison, the JSON expected objects are manually created (for simple objects) or parsed from the corresponding
/// JSON file to be found in test/aux/JsonOutputTest (for more complex objects).

using nlohmann::json;

// TEST(JsonOutputTest, ArithmeticExpr) { /* NOLINT */
//   auto lintValue = 22;
//   auto varIdentifier = "x";
//   // x + 22;
//   auto *aexp = new ArithmeticExpr(
//       new Variable(varIdentifier),
//       ArithmeticOp::ADDITION,
//       new LiteralInt(lintValue));
//   json j = {{"type", "ArithmeticExpr"},
//             {"leftOperand", {
//                 {"type", "Variable"},
//                 {"identifier", varIdentifier}}},
//             {"operator", OpSymb::getTextRepr(ArithmeticOp::ADDITION)},
//             {"rightOperand", {
//                 {"type", "LiteralInt"},
//                 {"value", lintValue}
//             }}};
//   EXPECT_EQ(aexp->toJson(), j);
// }
//
// TEST(JsonOutputTest, Return) { /* NOLINT */
//   // Return x > 22;
//   auto ret = new Return(
//       new LogicalExpr(new Variable("x"), LogCompOp::GREATER, 22));
//   json j = {
//       {"type", "Return"},
//       {"values", {{
//                       {"type", "LogicalExpr"},
//                       {"leftOperand", {
//                           {"type", "Variable"},
//                           {"identifier", "x"}
//                       }},
//                       {"rightOperand", {
//                           {"type", "LiteralInt"},
//                           {"value", 22}
//                       }},
//                       {"operator", ">"}}}
//       }};
//   EXPECT_EQ(ret->toJson(), j);
// }
//
// TEST(JsonOutputTest, UnaryExpr) { /* NOLINT */
//   auto varIdentifier = "x";
//   // !x
//   auto unaryExp = new UnaryExpr(UnaryOp::NEGATION, new Variable(varIdentifier));
//   json j = {{"type", "UnaryExpr"},
//             {"operator", OpSymb::getTextRepr(UnaryOp::NEGATION)},
//             {"rightOperand", {
//                 {"type", "Variable"},
//                 {"identifier", varIdentifier}}
//             }};
//   EXPECT_EQ(unaryExp->toJson(), j);
// }
//
// TEST(JsonOutputTest, FunctionParameter) { /* NOLINT */
//   auto fp = new FunctionParameter("int", new Variable("y"));
//   json j = {{"type", "FunctionParameter"},
//             {"datatype", {
//                 {"type", "Datatype"},
//                 {"specifier", "int"},
//                 {"encrypted", false}
//             }},
//             {"value", {
//                 {"type", "Variable"},
//                 {"identifier", "y"}
//             }}};
//   EXPECT_EQ(fp->toJson(), j);
// }
//
// TEST(JsonOutputTest, LogicalExpr) { /* NOLINT */
//   auto varIdentifier = "numIterations";
//   auto varIdentifierMax = "maxIterations";
//   // numIterations < maxIterations
//   auto *lexp = new LogicalExpr(
//       new Variable(varIdentifier),
//       LogCompOp::SMALLER,
//       new Variable(varIdentifierMax));
//   json j = {{"type", "LogicalExpr"},
//             {"leftOperand", {
//                 {"type", "Variable"},
//                 {"identifier", varIdentifier}}},
//             {"operator", OpSymb::getTextRepr(LogCompOp::SMALLER)},
//             {"rightOperand", {
//                 {"type", "Variable"},
//                 {"identifier", varIdentifierMax}
//             }}};
//   EXPECT_EQ(lexp->toJson(), j);
// }
//
// TEST(JsonOutputTest, Operator) { /* NOLINT */
//   auto opSub = new Operator(ArithmeticOp::SUBTRACTION);
//   EXPECT_EQ(opSub->getOperatorString(), OpSymb::getTextRepr(ArithmeticOp::SUBTRACTION));
//
//   auto opAnd = new Operator(LogCompOp::LOGICAL_AND);
//   EXPECT_EQ(opAnd->getOperatorString(), OpSymb::getTextRepr(LogCompOp::LOGICAL_AND));
// }
//
// TEST(JsonOutputTest, Block) { /* NOLINT */
//   auto bl = new Block(new VarDecl("width", Types::INT, new LiteralInt(22)));
//   json j = {{"type", "Block"},
//             {"statements", {{
//                                 {"type", "VarDecl"},
//                                 {"datatype", {
//                                     {"type", "Datatype"},
//                                     {"specifier", "int"},
//                                     {"encrypted", false}
//                                 }},
//                                 {"identifier", "width"},
//                                 {"initializer", {
//                                     {"type", "LiteralInt"},
//                                     {"value", 22}}}}}}};
//   EXPECT_EQ(bl->toJson(), j);
// }
//
// TEST(JsonOutputTest, CallExternal) { /* NOLINT */
//   std::vector<FunctionParameter *> funcParams = {
//       new FunctionParameter("string", new LiteralString("Hello world!"))};
//
//   // printf("Hello world!");
//   auto callExt = new CallExternal("printf", funcParams);
//
//   // read expected output from hard-coded file
//   std::ifstream f("../../test/expected_output_large/JsonOutputTest/CallExternal.json");
//   json j = json::parse(f);
//
//   EXPECT_EQ(callExt->toJson().dump(), j.dump());
// }
//
// TEST(JsonOutputTest, Call) {/* NOLINT */
//   // computeSecret(33)  --call to-->  int computeSecret(int inputA) { return inputA * 32; }
//   auto call = new Call(
//       {new FunctionParameter("int", new LiteralInt(33))},
//       new Function("computeSecret",
//                    {new FunctionParameter("int", new Variable("inputA"))},
//                    {new Return(
//                        new ArithmeticExpr(
//                            new Variable("inputA"),
//                            ArithmeticOp::MULTIPLICATION,
//                            new LiteralInt(32)))
//                    }));
//
//   // retrieve expected result
//   std::ifstream f("../../test/expected_output_large/JsonOutputTest/Call.json");
//   json expected = json::parse(f);
//
//   EXPECT_EQ(call->toJson().dump(), expected.dump());
// }
//
// TEST(JsonOutputTest, Function) { /* NOLINT */
//   // int computeX(int a, bool b, int z) {
//   //    return [(a + 221) * b] + z
//   // }
//   auto func = new Function("computeX", {
//       new FunctionParameter("int", new Variable("a")),
//       new FunctionParameter("bool", new Variable("b")),
//       new FunctionParameter("int", new Variable("z"))
//   }, {
//                                new Return(
//                                    new ArithmeticExpr(
//                                        new ArithmeticExpr(
//                                            new ArithmeticExpr(
//                                                new Variable("a"),
//                                                ArithmeticOp::ADDITION,
//                                                new LiteralInt(221)),
//                                            ArithmeticOp::MULTIPLICATION,
//                                            new Variable("b")),
//                                        ArithmeticOp::ADDITION,
//                                        new Variable("z")))});
//
//   // retrieve expected result
//   std::ifstream f("../../test/expected_output_large/JsonOutputTest/Function.json");
//   json expected = json::parse(f);
//
//   EXPECT_EQ(func->toJson(), expected);
// }
//
// TEST(JsonOutputTest, IfThenOnly) { /* NOLINT */
//   // if (b == true && z >= 17) {
//   //    isValid = true;
//   // }
//   auto ifStmt = new If(
//       new LogicalExpr(
//           new LogicalExpr(
//               new Variable("b"),
//               LogCompOp::EQUAL,
//               new LiteralBool(true)),
//           LogCompOp::LOGICAL_AND,
//           new LogicalExpr(
//               new Variable("z"),
//               LogCompOp::GREATER_EQUAL,
//               new LiteralInt(17))),
//       new VarAssignm("isValid", new LiteralBool(true)));
//
//   // retrieve expected result
//   std::ifstream f("../../test/expected_output_large/JsonOutputTest/IfThenOnly.json");
//   json expected = json::parse(f);
//
//   EXPECT_EQ(ifStmt->toJson(), expected);
// }
//
// TEST(JsonOutputTest, If) { /* NOLINT */
//   // if (b == true && z >= 17) {
//   //    isValid = true;
//   // } else {
//   //    isValid = false;
//   //    c = a-z;
//   // }
//
//   auto condition = new LogicalExpr(
//       new LogicalExpr(
//           new Variable("b"),
//           LogCompOp::EQUAL,
//           new LiteralBool(true)),
//       LogCompOp::LOGICAL_AND,
//       new LogicalExpr(
//           new Variable("z"),
//           LogCompOp::GREATER_EQUAL,
//           new LiteralInt(17)));
//
//   auto thenBranch = new VarAssignm("isValid", new LiteralBool(true));
//
//   std::vector<AbstractStatement *> elseStatement = {
//       new VarAssignm("isValid", new LiteralBool(false)),
//       new VarAssignm("c", new ArithmeticExpr(new Variable("a"), ArithmeticOp::SUBTRACTION, new Variable("z")))
//   };
//   auto *elseBranch = new Block(elseStatement);
//   auto ifStmt = new If(condition, thenBranch, elseBranch);
//
//   // retrieve expected result
//   std::ifstream f("../../test/expected_output_large/JsonOutputTest/If.json");
//   json expected = json::parse(f);
//
//   EXPECT_EQ(ifStmt->toJson(), expected);
// }
//
// TEST(JsonOutputTest, While) { /* NOLINT */
//   // while (i < 10) {
//   //   z = z*i;
//   //   !i;
//   // }
//   std::vector<AbstractStatement *> blockStatements;
//   blockStatements.emplace_back(
//       new VarAssignm("z",
//                      new ArithmeticExpr(
//                          new Variable("z"),
//                          ArithmeticOp::MULTIPLICATION,
//                          new Variable("i"))));
//   blockStatements.emplace_back(
//       new VarAssignm("i", new UnaryExpr(UnaryOp::NEGATION, new Variable("i"))));
//
//   auto whileStmt = new While(
//       new LogicalExpr(new Variable("i"), LogCompOp::SMALLER, new LiteralInt(10)),
//       new Block(blockStatements));
//
//   // retrieve expected result
//   std::ifstream f("../../test/expected_output_large/JsonOutputTest/While.json");
//   json expected = json::parse(f);
//
//   EXPECT_EQ(whileStmt->toJson(), expected);
// }
//
// TEST(JsonOutputTest, For) { /* NOLINT */
//   // for (int v = 42; v > 0; v = v-1) {
//   //   a = (a+b)/2;
//   // }
//   // int = 0;
//   auto forInitializer = new VarDecl("v", Types::INT, new LiteralInt(42));
//   // v > 0
//   auto forCondition = new LogicalExpr(new Variable("v"), GREATER, new LiteralInt(0));
//   // v = v-1
//   auto forUpdate = new VarAssignm("v",
//                                   new ArithmeticExpr(
//                                       new Variable("v"),
//                                       SUBTRACTION,
//                                       new LiteralInt(1)));
//   // sum = sum + base * i;
//   auto forBody = new Block(
//       new VarAssignm("a",
//                      new ArithmeticExpr(
//                          new ArithmeticExpr(
//                              new Variable("a"),
//                              ADDITION,
//                              new Variable("b")),
//                          DIVISION,
//                          new LiteralInt(2))));
//
//   auto forStmt = new For(forInitializer, forCondition, forUpdate, forBody);
//
//   // retrieve expected result
//   std::ifstream f("../../test/expected_output_large/JsonOutputTest/For.json");
//   json expected = json::parse(f);
//
//   EXPECT_EQ(forStmt->toJson(), expected);
// }
