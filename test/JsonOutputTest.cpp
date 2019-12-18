#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"
#include "VarAssignm.h"
#include "Variable.h"
#include "VarDecl.h"
#include "Block.h"
#include "BinaryExpr.h"
#include "LogicalExpr.h"
#include "Return.h"
#include "Group.h"
#include "UnaryExpr.h"
#include "CallExternal.h"
#include "FunctionParameter.h"

#include "gtest/gtest.h"


TEST(JsonOutputTest, LiteralInt) { /* NOLINT */
    int val = 2;
    auto *lint = new LiteralInt(val);
    json j = {{"type",  "LiteralInt"},
              {"value", val}};
    EXPECT_EQ(lint->toString(), j.dump());
}

TEST(JsonOutputTest, LiteralBool) { /* NOLINT */
    bool val = true;
    auto *lbool = new LiteralBool(val);
    json j = {{"type",  "LiteralBool"},
              {"value", val}};
    EXPECT_EQ(lbool->toString(), j.dump());
}

TEST(JsonOutputTest, LiteralString) { /* NOLINT */
    std::string val = "hello world!";
    auto *lString = new LiteralString(val);
    json j = {{"type",  "LiteralString"},
              {"value", val}};
    EXPECT_EQ(lString->toString(), j.dump());
}

TEST(JsonOutputTest, VarAssignm) { /* NOLINT */
    std::string identifier = "myCustomVar";
    int val = 2;
    auto *lint = new LiteralInt(val);
    // myCustomVar = 2;
    auto *assignm = new VarAssignm(identifier, lint);
    json j = {{"type",       "VarAssignm"},
              {"identifier", identifier},
              {"value",      {
                                     {"type", "LiteralInt"},
                                     {"value", val}
                             }
              }};
    EXPECT_EQ(assignm->toString(), j.dump());
}

TEST(JsonOutputTest, Variable) { /* NOLINT */
    std::string varIdentifier = "myVar";
    auto var = new Variable(varIdentifier);
    json j = {{"type",       "Variable"},
              {"identifier", varIdentifier}};
    EXPECT_EQ(var->toString(), j.dump());
}


TEST(JsonOutputTest, VarDecl) { /* NOLINT */
    auto identifier = "numIterations";
    auto datatype = "int";
    auto initializer = 3;
    // int numIterations = 3;
    auto *var = new VarDecl(identifier, datatype, initializer);
    json j = {{"type",        "VarDecl"},
              {"identifier",  identifier},
              {"datatype",    datatype},
              {"initializer", {
                                      {"type", "LiteralInt"},
                                      {"value", initializer}}
              }
    };
    EXPECT_EQ(var->toString(), j.dump());
}


TEST(JsonOutputTest, BinaryExpr) { /* NOLINT */
    auto lintValue = 22;
    auto varIdentifier = "x";
    // x + 22;
    auto *bexp = new BinaryExpr(
            new Variable(varIdentifier),
            OpSymb::BinaryOp::addition,
            new LiteralInt(lintValue)
    );
    json j = {{"type",         "BinaryExpr"},
              {"leftOperand",  {
                                       {"type", "Variable"},
                                       {"identifier", varIdentifier}}},
              {"operator",     OpSymb::getTextRepr(OpSymb::addition)},
              {"rightOperand", {
                                       {"type", "LiteralInt"},
                                       {"value",      lintValue}
                               }}};
    EXPECT_EQ(bexp->toString(), j.dump());
}


TEST(JsonOutputTest, Return) { /* NOLINT */
    // Return x > 22;
    auto ret = new Return(
            new LogicalExpr(new Variable("x"), OpSymb::greater, 22));
    json j = {
            {"type",  "Return"},
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
    EXPECT_EQ(ret->toString(), j.dump());
}

TEST(JsonOutputTest, UnaryExpr) { /* NOLINT */
    auto varIdentifier = "x";
    // !x
    auto unaryExp = new UnaryExpr(OpSymb::UnaryOp::negation, new Variable(varIdentifier));
    json j = {{"type",         "UnaryExpr"},
              {"operator",     OpSymb::getTextRepr(OpSymb::negation)},
              {"rightOperand", {
                                       {"type", "Variable"},
                                       {"identifier", varIdentifier}}
              }};
    EXPECT_EQ(unaryExp->toString(), j.dump());
}


TEST(JsonOutputTest, FunctionParameter) { /* NOLINT */
    auto fp = new FunctionParameter("int", new Variable("y"));
    json j = {{"type",     "FunctionParameter"},
              {"datatype", "int"},
              {"value",    {
                                   {"type", "Variable"},
                                   {"identifier", "y"}
                           }}};
    EXPECT_EQ(fp->toString(), j.dump());
}

TEST(JsonOutputTest, Group) { /* NOLINT */
    auto lintValue = 9883;
    auto varIdentifier = "totalItems";
    // (totalItems / 9883)
    auto gp = new Group(new BinaryExpr(
            new Variable(varIdentifier),
            OpSymb::BinaryOp::division,
            new LiteralInt(lintValue)
    ));

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
    EXPECT_EQ(gp->toString(), j.dump());
}


TEST(JsonOutputTest, LogicalExpr) { /* NOLINT */
    auto varIdentifier = "numIterations";
    auto varIdentifierMax = "maxIterations";
    // numIterations < maxIterations
    auto *lexp = new LogicalExpr(
            new Variable(varIdentifier),
            OpSymb::smaller,
            new Variable(varIdentifierMax)
    );
    json j = {{"type",         "LogicalExpr"},
              {"leftOperand",  {
                                       {"type", "Variable"},
                                       {"identifier", varIdentifier}}},
              {"operator",     OpSymb::getTextRepr(OpSymb::smaller)},
              {"rightOperand", {
                                       {"type", "Variable"},
                                       {"identifier", varIdentifierMax}
                               }}};
    EXPECT_EQ(lexp->toString(), j.dump());
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
    auto bl = new Block(new VarDecl("width", "int", new LiteralInt(22)));
    json j = {{"type",       "Block"},
              {"statements", {{
                                      {"type", "VarDecl"},
                                      {"identifier", "width"},
                                      {"datatype", "int"},
                                      {"initializer", {
                                                              {"type", "LiteralInt"},
                                                              {"value", 22}}}}}}};
    EXPECT_EQ(bl->toString(), j.dump());
}

TEST(JsonOutputTest, CallExternal) { /* NOLINT */
    // printf("Hello %s", "Oskar");
    auto *funcParams = new std::vector<FunctionParameter>;
    auto argOneType = "std::string";
    auto argOneStrValue = "Hello %s";
    funcParams->emplace_back(argOneType, new LiteralString(argOneStrValue));
    auto argTwoType = "const char*";
    auto argTwoIdentifier = "name";
    funcParams->emplace_back(argTwoType, new Variable(argTwoIdentifier)); // e.g. const char* name = "Oskar";
    auto funcName = "printf";
    auto callExt = new CallExternal(funcName, funcParams);
    json j = {{"type",         "CallExternal"},
              {"functionName", funcName},
              {"arguments",    {
                                       {{"type", "FunctionParameter"},
                                               {"datatype", argOneType},
                                               {"value", {
                                                                 {"type", "LiteralString"},
                                                                 {"value", argOneStrValue}
                                                         }
                                               }},
                                       {{"type", "FunctionParameter"},
                                               {"datatype", argTwoType},
                                               {"value", {
                                                                 {"type", "Variable"},
                                                                 {"identifier", argTwoIdentifier}}
                                               }
                                       }
                               }
              }};
    EXPECT_EQ(callExt->AbstractStatement::toString(), j.dump());
}

TEST(JsonOutputTest, Call) {/* NOLINT */
    // TODO implement me!
}

TEST(JsonOutputTest, Class) { /* NOLINT */
    // TODO implement me!
}

TEST(JsonOutputTest, Function) { /* NOLINT */
    // TODO implement me!
}

TEST(JsonOutputTest, If) { /* NOLINT */
    // TODO implement me!
}

TEST(JsonOutputTest, While) { /* NOLINT */
    // TODO implement me!
}
