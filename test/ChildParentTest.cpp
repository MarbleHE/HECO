#include "gtest/gtest.h"
#include "BinaryExpr.h"
#include "Variable.h"
#include "Return.h"
#include "Function.h"
#include "LiteralFloat.h"
#include "LogicalExpr.h"
#include "UnaryExpr.h"
#include "While.h"
#include "Block.h"
#include "Call.h"
#include "CallExternal.h"
#include "If.h"
#include "VarAssignm.h"
#include "VarDecl.h"

class BinaryExprFixture : public ::testing::Test {
 protected:
  LiteralInt* left;
  LiteralInt* otherLeft;
  LiteralFloat* right;
  LiteralFloat* otherRight;
  OpSymb::BinaryOp opSymb;
  Operator* operatorAdd;

  BinaryExprFixture() {
    left = new LiteralInt(3);
    otherLeft = new LiteralInt(42);
    right = new LiteralFloat(2.0);
    otherRight = new LiteralFloat(22.4);
    opSymb = OpSymb::addition;
    operatorAdd = new Operator(opSymb);
  }
};

TEST_F(BinaryExprFixture, BinaryExprStandardConstructor) {  /* NOLINT */
  auto* binaryExpr = new BinaryExpr(left, opSymb, right);

  // children
  ASSERT_EQ(binaryExpr->getChildren().size(), 3);
  ASSERT_EQ(binaryExpr->getChildAtIndex(0), left);
  ASSERT_TRUE(reinterpret_cast<Operator*>(binaryExpr->getChildAtIndex(1))->equals(opSymb));
  ASSERT_EQ(binaryExpr->getChildAtIndex(2), right);

  // parents
  ASSERT_EQ(binaryExpr->getParents().size(), 0);
  ASSERT_EQ(binaryExpr->getChildAtIndex(0)->getParents().size(), 1);
  ASSERT_TRUE(binaryExpr->getChildAtIndex(0)->hasParent(binaryExpr));
  ASSERT_EQ(binaryExpr->getChildAtIndex(1)->getParents().size(), 1);
  ASSERT_TRUE(binaryExpr->getChildAtIndex(1)->hasParent(binaryExpr));
  ASSERT_EQ(binaryExpr->getChildAtIndex(2)->getParents().size(), 1);
  ASSERT_TRUE(binaryExpr->getChildAtIndex(2)->hasParent(binaryExpr));
}

TEST_F(BinaryExprFixture, BinaryExprEmptyConstructor) {  /* NOLINT */
  BinaryExpr binaryExpr;
  ASSERT_EQ(binaryExpr.getChildren().size(), 3);
  ASSERT_EQ(binaryExpr.countChildrenNonNull(), 0);
  ASSERT_EQ(binaryExpr.getParents().size(), 0);
}

TEST_F(BinaryExprFixture, BinaryExprOperatorOnlyConstructor) {  /* NOLINT */
  auto* binaryExpr = new BinaryExpr(opSymb);

  // children
  ASSERT_EQ(binaryExpr->getChildren().size(), 3);
  ASSERT_EQ(binaryExpr->countChildrenNonNull(), 1);
  ASSERT_EQ(binaryExpr->getChildAtIndex(0), nullptr);
  ASSERT_TRUE(reinterpret_cast<Operator*>(binaryExpr->getChildAtIndex(1))->equals(opSymb));
  ASSERT_EQ(binaryExpr->getChildAtIndex(2), nullptr);

  // parents
  ASSERT_EQ(binaryExpr->getParents().size(), 0);
  ASSERT_EQ(binaryExpr->getChildAtIndex(1)->getParents().size(), 1);
  ASSERT_TRUE(binaryExpr->getChildAtIndex(1)->hasParent(binaryExpr));
}

TEST_F(BinaryExprFixture, BinaryExprAddChildException_NoEmptyChildSpotAvailable) {  /* NOLINT */
  auto* binaryExpr = new BinaryExpr(left, opSymb, right);
  EXPECT_THROW(binaryExpr->addChild(new LiteralInt(3)),
               std::logic_error);
}

TEST_F(BinaryExprFixture, BinaryExprAddChildException_TooManyChildrenAdded) {  /* NOLINT */
  auto* binaryExpr = new BinaryExpr(left, opSymb, right);
  EXPECT_THROW(binaryExpr->addChildren({{left, otherLeft, new Operator(opSymb), right}}, false),
               std::invalid_argument);
}

TEST_F(BinaryExprFixture, BinaryExprAddChildSuccess) {  /* NOLINT */
  auto* binaryExpr = new BinaryExpr();
  binaryExpr->setAttributes(nullptr, operatorAdd, right);
  auto newLeft = new LiteralInt(3);
  binaryExpr->addChildBilateral(newLeft);

  // children
  EXPECT_EQ(binaryExpr->getChildren().size(), 3);
  EXPECT_EQ(binaryExpr->getLeft(), newLeft);
  EXPECT_EQ(binaryExpr->getChildAtIndex(0), newLeft);
  EXPECT_EQ(binaryExpr->getOp(), operatorAdd);
  EXPECT_TRUE(reinterpret_cast<Operator*>(binaryExpr->getChildAtIndex(1))->equals(opSymb));
  EXPECT_EQ(binaryExpr->getRight(), right);
  EXPECT_EQ(binaryExpr->getChildAtIndex(2), right);

  // parents
  EXPECT_EQ(newLeft->getParents().size(), 1);
  EXPECT_EQ(newLeft->getParents().front(), binaryExpr);
  EXPECT_EQ(operatorAdd->getParents().size(), 1);
  EXPECT_EQ(operatorAdd->getParents().front(), binaryExpr);
  EXPECT_EQ(right->getParents().size(), 1);
  EXPECT_EQ(right->getParents().front(), binaryExpr);
}

TEST(ChildParentTests, Block) {  /* NOLINT */
  auto* blockStatement =
      new Block(new Call(
          {new FunctionParameter("int", new LiteralInt(22))},
          new Function("computeSecretNumber")));
  ASSERT_EQ(blockStatement->getChildren().size(), 0);
  ASSERT_EQ(blockStatement->getParents().size(), 0);
  ASSERT_FALSE(blockStatement->supportsCircuitMode());
  ASSERT_EQ(blockStatement->getMaxNumberChildren(), 0);
}

TEST(ChildParentTests, Call) {  /* NOLINT */
  auto* func = new Function("computeSecretX");
  auto* funcParam = new FunctionParameter(new Datatype(TYPES::INT), new LiteralInt(221));
  auto* call = new Call({funcParam}, func);

  // using AbstractExpr
  ASSERT_EQ(call->AbstractExpr::getChildren().size(), 0);
  ASSERT_EQ(call->AbstractExpr::getParents().size(), 0);
  ASSERT_FALSE(call->AbstractExpr::supportsCircuitMode());
  ASSERT_EQ(call->AbstractExpr::getMaxNumberChildren(), 0);

  // using AbstractStatement
  ASSERT_EQ(call->AbstractStatement::getChildren().size(), 0);
  ASSERT_EQ(call->AbstractStatement::getParents().size(), 0);
  ASSERT_FALSE(call->AbstractStatement::supportsCircuitMode());
  ASSERT_EQ(call->AbstractStatement::getMaxNumberChildren(), 0);

  // checking children
  ASSERT_EQ(func->getParents().size(), 0);
  ASSERT_EQ(funcParam->getParents().size(), 0);
}

TEST(ChildParentTests, CallExternal) {  /* NOLINT */
  auto* callExternal = new CallExternal("computeSecretKeys");

  // using AbstractExpr
  ASSERT_EQ(callExternal->AbstractExpr::getChildren().size(), 0);
  ASSERT_EQ(callExternal->AbstractExpr::getParents().size(), 0);
  ASSERT_FALSE(callExternal->AbstractExpr::supportsCircuitMode());
  ASSERT_EQ(callExternal->AbstractExpr::getMaxNumberChildren(), 0);

  // using AbstractStatement
  ASSERT_EQ(callExternal->AbstractStatement::getChildren().size(), 0);
  ASSERT_EQ(callExternal->AbstractStatement::getParents().size(), 0);
  ASSERT_FALSE(callExternal->AbstractStatement::supportsCircuitMode());
  ASSERT_EQ(callExternal->AbstractStatement::getMaxNumberChildren(), 0);
}

class FunctionFixture : public ::testing::Test {
 protected:
  Function* funcComputeX;
  AbstractStatement* returnStatement;

  FunctionFixture() {
    funcComputeX = new Function("computeX");
    returnStatement = new Return(new LiteralBool(true));
  }
};

TEST_F(FunctionFixture, FunctionAddStatement) {  /* NOLINT */
  funcComputeX->addStatement(returnStatement);
  ASSERT_EQ(funcComputeX->getChildren().size(), 0);
  ASSERT_EQ(returnStatement->getParents().size(), 0);
}

TEST_F(FunctionFixture, FunctionNotSupportedInCircuitMode) {  /* NOLINT */
  // Function is not circuit-compatible, i.e., does not support use of child/parent relationship
  ASSERT_THROW(funcComputeX->addChild(returnStatement), std::logic_error);
  ASSERT_EQ(funcComputeX->getChildren().size(), 0);
  ASSERT_EQ(returnStatement->getParents().size(), 0);
  ASSERT_FALSE(funcComputeX->supportsCircuitMode());
  ASSERT_EQ(funcComputeX->getMaxNumberChildren(), 0);
}

class FunctionParameterFixture : public ::testing::Test {
 protected:
  Datatype* datatype;
  Datatype* datatype2;
  TYPES datatypeEnum;
  std::string datatypeAsString;
  AbstractExpr* variableThreshold;
  AbstractExpr* variableSecret;

  FunctionParameterFixture() {
    datatypeEnum = TYPES::INT;
    datatypeAsString = Datatype::enum_to_string(datatypeEnum);
    datatype = new Datatype(datatypeEnum);
    datatype2 = new Datatype(TYPES::FLOAT);
    variableThreshold = new Variable("threshold");
    variableSecret = new Variable("secretNumber");
  }
};

TEST_F(FunctionParameterFixture, FunctionParameterStandardConstructor) {  /* NOLINT */
  auto* functionParameter = new FunctionParameter(datatypeAsString, variableThreshold);

  // children
  ASSERT_EQ(functionParameter->getChildren().size(), 2);
  ASSERT_EQ(functionParameter->getChildAtIndex(0)->castTo<Datatype>()->getType(), datatypeEnum);
  ASSERT_EQ(functionParameter->getChildAtIndex(1), variableThreshold);

  // parents
  ASSERT_EQ(functionParameter->getParents().size(), 0);
  ASSERT_EQ(functionParameter->getChildAtIndex(0)->getParents().size(), 1);
  ASSERT_TRUE(functionParameter->getChildAtIndex(0)->hasParent(functionParameter));
  ASSERT_EQ(functionParameter->getChildAtIndex(1)->getParents().size(), 1);
  ASSERT_TRUE(functionParameter->getChildAtIndex(1)->hasParent(functionParameter));
}

TEST_F(FunctionParameterFixture, FunctionParameterAddChildExceptionDatatypeConstructor) {  /* NOLINT */
  auto* functionParameter = new FunctionParameter(datatype, variableThreshold);

  // children
  ASSERT_EQ(functionParameter->getChildren().size(), 2);
  ASSERT_EQ(functionParameter->getChildAtIndex(0), datatype);
  ASSERT_EQ(functionParameter->getChildAtIndex(1), variableThreshold);

  // parents
  ASSERT_EQ(functionParameter->getParents().size(), 0);
  ASSERT_EQ(functionParameter->getChildAtIndex(0)->getParents().size(), 1);
  ASSERT_TRUE(functionParameter->getChildAtIndex(0)->hasParent(functionParameter));
  ASSERT_EQ(functionParameter->getChildAtIndex(1)->getParents().size(), 1);
  ASSERT_TRUE(functionParameter->getChildAtIndex(1)->hasParent(functionParameter));
}

TEST_F(FunctionParameterFixture, FunctionParameterAddChildException_NoEmptyChildSpotAvailable) {  /* NOLINT */
  auto* functionParameter = new FunctionParameter(datatype, variableThreshold);
  EXPECT_THROW(functionParameter->addChild(variableSecret), std::logic_error);
}

TEST_F(FunctionParameterFixture, FunctionParameterAddChildException_TooManyChildrenAdded) {  /* NOLINT */
  auto* functionParameter = new FunctionParameter(datatype, variableThreshold);
  EXPECT_THROW(functionParameter->addChildren({{datatype, variableSecret, variableThreshold}}), std::invalid_argument);
}

TEST_F(FunctionParameterFixture, FunctionParameter_AddChildSuccess) {  /* NOLINT */
  auto* functionParameter = new FunctionParameter(datatype, variableThreshold);

  functionParameter->removeChild(variableThreshold);
  functionParameter->addChildBilateral(variableSecret);

  // children
  EXPECT_EQ(functionParameter->getChildren().size(), 2);
  EXPECT_EQ(functionParameter->getValue(), variableSecret);
  EXPECT_EQ(functionParameter->getChildAtIndex(1), variableSecret);

  // parents
  EXPECT_EQ(functionParameter->getParents().size(), 0);
  EXPECT_EQ(variableSecret->getParents().size(), 1);
  EXPECT_EQ(variableSecret->getParents().front(), functionParameter);
  EXPECT_TRUE(variableSecret->hasParent(functionParameter));

  functionParameter->removeChild(datatype);
  functionParameter->addChildBilateral(datatype2);

  // children
  EXPECT_EQ(functionParameter->getChildren().size(), 2);
  EXPECT_EQ(functionParameter->getDatatype(), datatype2);
  EXPECT_EQ(functionParameter->getChildAtIndex(0), datatype2);

  // parents
  EXPECT_EQ(functionParameter->getParents().size(), 0);
  EXPECT_EQ(datatype2->getParents().size(), 1);
  EXPECT_EQ(datatype2->getParents().front(), functionParameter);
  EXPECT_TRUE(datatype2->hasParent(functionParameter));
}

TEST(ChildParentTests, If) {  /* NOLINT */
  auto ifStatement = new If(
      new LogicalExpr(new LiteralInt(33), OpSymb::greater, new CallExternal("computeX")),
      new Block(),
      new Block());
  ASSERT_EQ(ifStatement->AbstractStatement::getChildren().size(), 0);
  ASSERT_EQ(ifStatement->AbstractStatement::getParents().size(), 0);
  ASSERT_FALSE(ifStatement->AbstractStatement::supportsCircuitMode());
  ASSERT_EQ(ifStatement->AbstractStatement::getMaxNumberChildren(), 0);
}

TEST(ChildParentTests, LiteralBoolHasNoChildrenOrParents) {  /* NOLINT */
  // Literals should never have any children
  LiteralBool literalBool(true);
  ASSERT_TRUE(literalBool.getChildren().empty());
  ASSERT_TRUE(literalBool.getParents().empty());

  literalBool.setValue(false);
  ASSERT_TRUE(literalBool.getChildren().empty());
  ASSERT_TRUE(literalBool.getParents().empty());
}

TEST(ChildParentTests, LiteralFloatHasNoChildrenOrParents) {  /* NOLINT */
  // Literals should never have any children
  LiteralFloat literalFloat(true);
  ASSERT_TRUE(literalFloat.getChildren().empty());
  ASSERT_TRUE(literalFloat.getParents().empty());

  literalFloat.setValue(false);
  ASSERT_TRUE(literalFloat.getChildren().empty());
  ASSERT_TRUE(literalFloat.getParents().empty());
}

TEST(ChildParentTests, LiteralIntHasNoChildrenOrParents) {  /* NOLINT */
  // Literals should never have any children
  LiteralInt literalInt(33);
  ASSERT_TRUE(literalInt.getChildren().empty());
  ASSERT_TRUE(literalInt.getParents().empty());

  literalInt.setValue(111);
  ASSERT_TRUE(literalInt.getChildren().empty());
  ASSERT_TRUE(literalInt.getParents().empty());
}

TEST(ChildParentTests, LiteralStringHasNoChildrenOrParents) {  /* NOLINT */
  // Literals should never have any children
  LiteralString literalString("alpha");
  ASSERT_TRUE(literalString.getChildren().empty());
  ASSERT_TRUE(literalString.getParents().empty());

  literalString.setValue("beta");
  ASSERT_TRUE(literalString.getChildren().empty());
  ASSERT_TRUE(literalString.getParents().empty());
}

class LogicalExprFixture : public ::testing::Test {
 protected:
  LiteralInt* literalInt;
  LiteralInt* literalIntAnother;
  LiteralBool* literalBool;
  OpSymb::LogCompOp opSymb;
  Operator* operatorGreaterEqual;

  LogicalExprFixture() {
    literalInt = new LiteralInt(24);
    literalIntAnother = new LiteralInt(6245);
    literalBool = new LiteralBool(true);
    opSymb = OpSymb::greaterEqual;
    operatorGreaterEqual = new Operator(opSymb);
  }
};

TEST_F(LogicalExprFixture, LogicalExprStandardConstructor) {  /* NOLINT */
  auto* logicalExpr = new LogicalExpr(literalInt, opSymb, literalIntAnother);

  // children
  ASSERT_EQ(logicalExpr->getChildren().size(), 3);
  ASSERT_EQ(logicalExpr->getChildAtIndex(0), literalInt);
  ASSERT_TRUE(reinterpret_cast<Operator*>(logicalExpr->getChildAtIndex(1))->equals(opSymb));
  ASSERT_EQ(logicalExpr->getChildAtIndex(2), literalIntAnother);

  // parents
  ASSERT_EQ(logicalExpr->getParents().size(), 0);
  ASSERT_EQ(logicalExpr->getChildAtIndex(0)->getParents().size(), 1);
  ASSERT_TRUE(logicalExpr->getChildAtIndex(0)->hasParent(logicalExpr));
  ASSERT_EQ(logicalExpr->getChildAtIndex(1)->getParents().size(), 1);
  ASSERT_TRUE(logicalExpr->getChildAtIndex(1)->hasParent(logicalExpr));
  ASSERT_EQ(logicalExpr->getChildAtIndex(2)->getParents().size(), 1);
  ASSERT_TRUE(logicalExpr->getChildAtIndex(2)->hasParent(logicalExpr));
}

TEST_F(LogicalExprFixture, LogicalExprEmptyConstructor) {  /* NOLINT */
  BinaryExpr logicalExpr;
  ASSERT_EQ(logicalExpr.getChildren().size(), 3);
  ASSERT_EQ(logicalExpr.countChildrenNonNull(), 0);
  ASSERT_EQ(logicalExpr.getParents().size(), 0);
}

TEST_F(LogicalExprFixture, LogicalExprOperatorOnlyConstructor) {  /* NOLINT */
  auto* logicalExpr = new LogicalExpr(opSymb);

  // children
  ASSERT_EQ(logicalExpr->getChildren().size(), 3);
  ASSERT_EQ(logicalExpr->countChildrenNonNull(), 1);
  ASSERT_EQ(logicalExpr->getChildAtIndex(0), nullptr);
  ASSERT_TRUE(reinterpret_cast<Operator*>(logicalExpr->getChildAtIndex(1))->equals(opSymb));
  ASSERT_EQ(logicalExpr->getChildAtIndex(2), nullptr);

  // parents
  ASSERT_EQ(logicalExpr->getParents().size(), 0);
  ASSERT_EQ(logicalExpr->getChildAtIndex(1)->getParents().size(), 1);
  ASSERT_TRUE(logicalExpr->getChildAtIndex(1)->hasParent(logicalExpr));
}

TEST_F(LogicalExprFixture, LogicalExprAddChildException_NoEmptyChildSpotAvailable) {  /* NOLINT */
  auto* logicalExpr = new LogicalExpr(literalInt, opSymb, literalIntAnother);
  EXPECT_THROW(logicalExpr->addChild(new LiteralInt(3)),
               std::logic_error);
}

TEST_F(LogicalExprFixture, LogicalExprAddChildException_TooManyChildrenAdded) {  /* NOLINT */
  auto* logicalExpr = new LogicalExpr(literalInt, opSymb, literalIntAnother);
  EXPECT_THROW(logicalExpr->addChildren({{literalInt, literalIntAnother, new Operator(opSymb), literalBool}}, false),
               std::invalid_argument);
}

TEST_F(LogicalExprFixture, LogicalExprAddChildSuccess) {  /* NOLINT */
  auto* logicalExpr = new LogicalExpr();
  logicalExpr->setAttributes(nullptr, operatorGreaterEqual, literalBool);
  logicalExpr->addChildBilateral(literalIntAnother);

  // children
  EXPECT_EQ(logicalExpr->getChildren().size(), 3);
  EXPECT_EQ(logicalExpr->getLeft(), literalIntAnother);
  EXPECT_EQ(logicalExpr->getChildAtIndex(0), literalIntAnother);
  EXPECT_EQ(logicalExpr->getOp(), operatorGreaterEqual);
  EXPECT_TRUE(reinterpret_cast<Operator*>(logicalExpr->getChildAtIndex(1))->equals(opSymb));
  EXPECT_EQ(logicalExpr->getRight(), literalBool);
  EXPECT_EQ(logicalExpr->getChildAtIndex(2), literalBool);

  // parents
  EXPECT_EQ(literalIntAnother->getParents().size(), 1);
  EXPECT_EQ(literalIntAnother->getParents().front(), logicalExpr);
  EXPECT_EQ(operatorGreaterEqual->getParents().size(), 1);
  EXPECT_EQ(operatorGreaterEqual->getParents().front(), logicalExpr);
  EXPECT_EQ(literalBool->getParents().size(), 1);
  EXPECT_EQ(literalBool->getParents().front(), logicalExpr);
}

TEST(ChildParentTests, OperatorHasNoChildrenOrParents) {  /* NOLINT */
  Operator op(OpSymb::greaterEqual);
  ASSERT_TRUE(op.getChildren().empty());
  ASSERT_TRUE(op.getParents().empty());
}

class ReturnStatementFixture : public ::testing::Test {
 protected:
  AbstractExpr* abstractExpr;
  AbstractExpr* abstractExprOther;

  ReturnStatementFixture() {
    abstractExpr = new LiteralInt(22);
    abstractExprOther = new LiteralBool(true);
  }
};

TEST_F(ReturnStatementFixture, ReturnStatementStandardConstructor) {  /* NOLINT */
  auto returnStatement = new Return(abstractExpr);

  // children
  ASSERT_EQ(returnStatement->getChildren().size(), 1);
  ASSERT_EQ(returnStatement->getChildren().front(), abstractExpr);

  // parent
  ASSERT_EQ(returnStatement->getParents().size(), 0);
  ASSERT_EQ(returnStatement->getChildAtIndex(0)->getParents().size(), 1);
  ASSERT_TRUE(returnStatement->getChildAtIndex(0)->hasParent(returnStatement));
}

TEST_F(ReturnStatementFixture, ReturnStatementEmptyConstructor) {  /* NOLINT */
  Return returnStatement;
  ASSERT_EQ(returnStatement.getChildren().size(), 1);
  ASSERT_EQ(returnStatement.countChildrenNonNull(), 0);
  ASSERT_EQ(returnStatement.getParents().size(), 0);
}

TEST_F(ReturnStatementFixture, ReturnStatementAddChildException_NoEmptyChildSpotAvailable) {  /* NOLINT */
  auto* returnStatement = new Return(abstractExpr);
  EXPECT_THROW(returnStatement->addChild(abstractExprOther),
               std::logic_error);
}

TEST_F(ReturnStatementFixture, ReturnStatementAddChildException_TooManyChildrenAdded) {  /* NOLINT */
  auto* returnStatement = new Return(abstractExpr);
  EXPECT_THROW(returnStatement->addChildren({{abstractExpr, abstractExprOther}}, false),
               std::invalid_argument);
}

TEST_F(ReturnStatementFixture, ReturnStatementAddChildSuccess) {  /* NOLINT */
  auto* returnStatement = new Return();
  returnStatement->addChildBilateral(abstractExprOther);
  EXPECT_EQ(returnStatement->getReturnExpr(), abstractExprOther);
  EXPECT_EQ(returnStatement->getChildren().size(), 1);
  EXPECT_EQ(returnStatement->getChildren().front(), abstractExprOther);
  EXPECT_EQ(abstractExprOther->getParents().size(), 1);
  EXPECT_EQ(abstractExprOther->getParents().front(), returnStatement);
}

class UnaryExprFixture : public ::testing::Test {
 protected:
  OpSymb::UnaryOp opSymbNegation;
  LiteralBool* literalBoolTrue;

  UnaryExprFixture() {
    opSymbNegation = OpSymb::negation;
    new Operator(opSymbNegation);
    literalBoolTrue = new LiteralBool(true);
  }
};

TEST_F(UnaryExprFixture, UnaryExprStandardConstructor) {  /* NOLINT */
  auto* unaryExpr = new UnaryExpr(opSymbNegation, literalBoolTrue);

  // children
  ASSERT_EQ(unaryExpr->getChildren().size(), 2);
  ASSERT_TRUE(reinterpret_cast<Operator*>(unaryExpr->getChildAtIndex(0))->equals(opSymbNegation));
  ASSERT_EQ(unaryExpr->getChildAtIndex(1), literalBoolTrue);

  // parents
  ASSERT_EQ(unaryExpr->getParents().size(), 0);
  ASSERT_TRUE(unaryExpr->getChildAtIndex(0)->hasParent(unaryExpr));
  ASSERT_TRUE(unaryExpr->getChildAtIndex(1)->hasParent(unaryExpr));
}

TEST_F(UnaryExprFixture, UnaryExprAddChildException_NoEmptyChildSpotAvailable) {  /* NOLINT */
  auto* unaryExpr = new UnaryExpr(opSymbNegation, literalBoolTrue);
  EXPECT_THROW(unaryExpr->addChild(new Operator(OpSymb::decrement)), std::logic_error);
}

TEST_F(UnaryExprFixture, UnaryExprAddChildException_TooManyChildrenAdded) {  /* NOLINT */
  auto* unaryExpr = new UnaryExpr(opSymbNegation, literalBoolTrue);
  EXPECT_THROW(unaryExpr->addChildren({new Operator(OpSymb::decrement), new LiteralBool(false)}), std::logic_error);
}

TEST_F(UnaryExprFixture, UnaryExprtion_AddChildSuccess) {  /* NOLINT */
  auto* unaryExpr = new UnaryExpr(opSymbNegation, literalBoolTrue);

  unaryExpr->removeChild(unaryExpr->getOp());
  auto* newOperator = new Operator(OpSymb::decrement);
  unaryExpr->addChildBilateral(newOperator);

  // children
  EXPECT_EQ(unaryExpr->getChildren().size(), 2);
  EXPECT_EQ(*unaryExpr->getOp(), *newOperator);
  EXPECT_TRUE(reinterpret_cast<Operator*>(unaryExpr->getChildAtIndex(0))->equals(newOperator->getOperatorSymbol()));

  // parents
  EXPECT_EQ(unaryExpr->getParents().size(), 0);
  EXPECT_EQ(unaryExpr->getChildAtIndex(0)->getParents().size(), 1);
  EXPECT_TRUE(unaryExpr->getChildAtIndex(0)->hasParent(unaryExpr));
}

class VarAssignmFixture : public ::testing::Test {
 protected:
  LiteralInt* literalInt222;
  std::string variableIdentifier;

  VarAssignmFixture() {
    literalInt222 = new LiteralInt(222);
    variableIdentifier = "secretX";
  }
};

TEST_F(VarAssignmFixture, VarAssignmStandardConstructor) {  /* NOLINT */
  auto varAssignm = new VarAssignm(variableIdentifier, literalInt222);

  // children
  ASSERT_EQ(varAssignm->getChildren().size(), 1);
  ASSERT_EQ(varAssignm->getChildAtIndex(0), literalInt222);

  // parents
  ASSERT_EQ(varAssignm->getParents().size(), 0);
  ASSERT_EQ(varAssignm->getChildAtIndex(0)->getParents().size(), 1);
  ASSERT_TRUE(varAssignm->getChildAtIndex(0)->hasParent(varAssignm));
}

TEST_F(VarAssignmFixture, VarAssignm_NoEmptyChildSpotAvailable) {  /* NOLINT */
  auto varAssignm = new VarAssignm(variableIdentifier, literalInt222);
  EXPECT_THROW(varAssignm->addChild(new LiteralBool(true)), std::logic_error);
}

TEST_F(VarAssignmFixture, VarAssignm_TooManyChildrenAdded) {  /* NOLINT */
  auto varAssignm = new VarAssignm(variableIdentifier, literalInt222);
  EXPECT_THROW(varAssignm->addChildren({new LiteralBool(true), new LiteralInt(5343)}), std::invalid_argument);
}

TEST_F(VarAssignmFixture, VarAssignmAddChildSuccess) {  /* NOLINT */
  auto varAssignm = new VarAssignm(variableIdentifier, literalInt222);

  varAssignm->removeChildren();
  auto newChild = new LiteralBool(false);
  varAssignm->addChildBilateral(newChild);

  // children
  ASSERT_EQ(varAssignm->getChildren().size(), 1);
  ASSERT_EQ(varAssignm->getChildAtIndex(0), newChild);

  // parents
  ASSERT_EQ(varAssignm->getParents().size(), 0);
  ASSERT_EQ(varAssignm->getChildAtIndex(0)->getParents().size(), 1);
  ASSERT_TRUE(varAssignm->getChildAtIndex(0)->hasParent(varAssignm));
}

class VarDeclFixture : public ::testing::Test {
 protected:
  int integerValue;
  float floatValue;
  bool boolValue;
  std::string stringValue;
  LiteralInt* literalInt;
  std::string variableIdentifier;
  TYPES datatypeInt;

  VarDeclFixture() {
    literalInt = new LiteralInt(integerValue);
    integerValue = 343224;
    variableIdentifier = "maxValue";
    floatValue = 2.42f;
    boolValue = false;
    stringValue = "Determines the maximum allowed value";
    datatypeInt = TYPES::INT;
  }

  static void checkExpected(VarDecl* varDeclaration, Datatype* expectedDatatype, AbstractExpr* expectedValue) {
    // children
    ASSERT_EQ(varDeclaration->getChildren().size(), 2);
    ASSERT_EQ(reinterpret_cast<Datatype*>(varDeclaration->getChildAtIndex(0)), expectedDatatype);
    ASSERT_EQ(varDeclaration->getChildAtIndex(1), expectedValue);

    // parents
    ASSERT_EQ(varDeclaration->getParents().size(), 0);
    ASSERT_EQ(varDeclaration->getChildAtIndex(0)->getParents().size(), 1);
    ASSERT_TRUE(varDeclaration->getChildAtIndex(0)->hasParent(varDeclaration));
    ASSERT_EQ(varDeclaration->getChildAtIndex(1)->getParents().size(), 1);
    ASSERT_TRUE(varDeclaration->getChildAtIndex(1)->hasParent(varDeclaration));
  }
};

TEST_F(VarDeclFixture, VarDeclStandardConstructor) {  /* NOLINT */
  auto* variableDeclaration = new VarDecl(variableIdentifier, datatypeInt, literalInt);
  ASSERT_EQ(reinterpret_cast<Datatype*>(variableDeclaration->getDatatype())->toString(),
            Datatype::enum_to_string(datatypeInt));
  checkExpected(variableDeclaration, variableDeclaration->getDatatype(), literalInt);
}

TEST_F(VarDeclFixture, VarDeclIntConstructor) {  /* NOLINT */
  auto* variableDeclaration = new VarDecl(variableIdentifier, integerValue);
  ASSERT_EQ(reinterpret_cast<LiteralInt*>(variableDeclaration->getInitializer())->getValue(), integerValue);
  ASSERT_EQ(reinterpret_cast<Datatype*>(variableDeclaration->getDatatype())->toString(),
            Datatype::enum_to_string(TYPES::INT));
  checkExpected(variableDeclaration, variableDeclaration->getDatatype(), variableDeclaration->getInitializer());
}

TEST_F(VarDeclFixture, VarDeclBoolConstructor) {  /* NOLINT */
  auto* variableDeclaration = new VarDecl(variableIdentifier, boolValue);
  ASSERT_EQ(reinterpret_cast<LiteralBool*>(variableDeclaration->getInitializer())->getValue(), boolValue);
  ASSERT_EQ(reinterpret_cast<Datatype*>(variableDeclaration->getDatatype())->toString(),
            Datatype::enum_to_string(TYPES::BOOL));
  checkExpected(variableDeclaration, variableDeclaration->getDatatype(), variableDeclaration->getInitializer());
}

TEST_F(VarDeclFixture, VarDeclFloatConstructor) {  /* NOLINT */
  auto* variableDeclaration = new VarDecl(variableIdentifier, floatValue);
  ASSERT_EQ(reinterpret_cast<LiteralFloat*>(variableDeclaration->getInitializer())->getValue(), floatValue);
  ASSERT_EQ(reinterpret_cast<Datatype*>(variableDeclaration->getDatatype())->toString(),
            Datatype::enum_to_string(TYPES::FLOAT));
  checkExpected(variableDeclaration, variableDeclaration->getDatatype(), variableDeclaration->getInitializer());
}

TEST_F(VarDeclFixture, VarDeclStringConstructor) {  /* NOLINT */
  auto* variableDeclaration = new VarDecl(variableIdentifier, stringValue);
  ASSERT_EQ(reinterpret_cast<LiteralString*>(variableDeclaration->getInitializer())->getValue(), stringValue);
  ASSERT_EQ(reinterpret_cast<Datatype*>(variableDeclaration->getDatatype())->toString(),
            Datatype::enum_to_string(TYPES::STRING));
  checkExpected(variableDeclaration, variableDeclaration->getDatatype(), variableDeclaration->getInitializer());
}

TEST(ChildParentTests, Variable) {  /* NOLINT */
  Variable variable("myInt");
  ASSERT_TRUE(variable.getChildren().empty());
  ASSERT_TRUE(variable.getParents().empty());
}

TEST(ChildParentTests, While) {  /* NOLINT */
  auto* whileStatement =
      new While(new LogicalExpr(new LiteralInt(32), OpSymb::greaterEqual, new Variable("a")), new Block());
  ASSERT_EQ(whileStatement->getChildren().size(), 0);
  ASSERT_EQ(whileStatement->getParents().size(), 0);
  ASSERT_FALSE(whileStatement->supportsCircuitMode());
  ASSERT_EQ(whileStatement->getMaxNumberChildren(), 0);
}
