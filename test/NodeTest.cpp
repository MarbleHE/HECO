#include "Call.h"
#include "CallExternal.h"
#include "AbstractNode.h"
#include "NodeUtils.h"
#include "Variable.h"
#include "LiteralBool.h"
#include "gtest/gtest.h"
#include "Function.h"

TEST(NodeTest, testUniqueNodeId_Call) {
  auto callNode = new Call(new Function("computeX"));
  ASSERT_EQ(callNode->AbstractExpr::getUniqueNodeId(), callNode->getUniqueNodeId());
}

TEST(NodeTest, testUniqueNodeId_CallExternal__UnfixedTest) {
  auto callExtNode = new CallExternal("genSecretKeys");
  ASSERT_EQ(callExtNode->AbstractExpr::getUniqueNodeId(), callExtNode->getUniqueNodeId());
}


