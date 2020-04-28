#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/CallExternal.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/LiteralBool.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/utilities/NodeUtils.h"
#include "gtest/gtest.h"

TEST(NodeTest, testUniqueNodeId_Call) {
  auto callNode = new Call(new Function("computeX"));
  ASSERT_EQ(callNode->AbstractExpr::getUniqueNodeId(), callNode->getUniqueNodeId());
}

TEST(NodeTest, testUniqueNodeId_CallExternal__UnfixedTest) {
  auto callExtNode = new CallExternal("genSecretKeys");
  ASSERT_EQ(callExtNode->AbstractExpr::getUniqueNodeId(), callExtNode->getUniqueNodeId());
}


