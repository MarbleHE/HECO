#include "Call.h"
#include "CallExternal.h"
#include "Node.h"
#include "Variable.h"
#include "LiteralBool.h"
#include "gtest/gtest.h"
#include "Function.h"

TEST(NodeTest, rewriteMultiInputGateToBinaryGatesChain_emptyInputs) { /* NOLINT */
  std::vector<Node *> inputs{};
  ASSERT_THROW(Node::rewriteMultiInputGateToBinaryGatesChain(inputs, OpSymb::logicalAnd), std::invalid_argument);
}

TEST(NodeTest, rewriteMultiInputGateToBinaryGatesChain_oneInputAND) { /* NOLINT */
  std::vector<Node *> inputs{new Variable("alpha")};
  OpSymb::LogCompOp gateType = OpSymb::logicalAnd;
  auto result = Node::rewriteMultiInputGateToBinaryGatesChain(inputs, gateType);

  // create new AST for evaluation
  Ast ast(result.back());

  // if alpha = false => true AND alpha = false
  auto resultEval = ast.evaluateCircuit(
      {{"alpha", new LiteralBool(false)}}, false).front();
  ASSERT_EQ(*resultEval, LiteralBool(false));

  // if alpha = true => true AND alpha = false
  resultEval = ast.evaluateCircuit(
      {{"alpha", new LiteralBool(true)}}, false).front();
  ASSERT_EQ(*resultEval, LiteralBool(true));
}

TEST(NodeTest, rewriteMultiInputGateToBinaryGatesChain_oneInputXOR) { /* NOLINT */
  std::vector<Node *> inputs{new Variable("alpha")};
  OpSymb::LogCompOp gateType = OpSymb::logicalXor;
  auto result = Node::rewriteMultiInputGateToBinaryGatesChain(inputs, gateType);

  // create new AST for evaluation
  Ast ast(result.back());

  // if alpha = false => false XOR alpha = false
  auto resultEval = ast.evaluateCircuit(
      {{"alpha", new LiteralBool(false)}}, false).front();
  ASSERT_EQ(*resultEval, LiteralBool(false));

  // if alpha = true => false XOR true = true
  resultEval = ast.evaluateCircuit(
      {{"alpha", new LiteralBool(true)}}, false).front();
  ASSERT_EQ(*resultEval, LiteralBool(true));
}

TEST(NodeTest, rewriteMultiInputGateToBinaryGatesChain_oneInputUnsupportedException) { /* NOLINT */
  std::vector<Node *> inputs{new Variable("alpha")};
  OpSymb::LogCompOp gateType = OpSymb::logicalOr;
  ASSERT_THROW(Node::rewriteMultiInputGateToBinaryGatesChain(inputs, gateType),
               std::runtime_error);
}

TEST(NodeTest, rewriteMultiInputGateToBinaryGatesChain_multipleInputs) { /* NOLINT */
  std::vector<Node *> inputs{new Variable("alpha"),
                             new Variable("beta"),
                             new Variable("gamma"),
                             new Variable("delta")};
  OpSymb::LogCompOp gateType = OpSymb::logicalXor;
  auto result = Node::rewriteMultiInputGateToBinaryGatesChain(inputs, gateType);

  // create new AST for evaluation
  Ast ast(result.back());

  // if all are false => evaluate to false
  auto resultEval = ast.evaluateCircuit(
      {{"alpha", new LiteralBool(false)},
       {"beta", new LiteralBool(false)},
       {"gamma", new LiteralBool(false)},
       {"delta", new LiteralBool(false)}},
      false).front();
  ASSERT_EQ(*resultEval, LiteralBool(false));

  // if exactly one is true => evaluate to true
  resultEval = ast.evaluateCircuit(
      {{"alpha", new LiteralBool(false)},
       {"beta", new LiteralBool(false)},
       {"gamma", new LiteralBool(true)},
       {"delta", new LiteralBool(false)}},
      false).front();
  ASSERT_EQ(*resultEval, LiteralBool(true));

  // if multiple are true => evaluate to false
  resultEval = ast.evaluateCircuit(
      {{"alpha", new LiteralBool(true)},
       {"beta", new LiteralBool(false)},
       {"gamma", new LiteralBool(true)},
       {"delta", new LiteralBool(false)}},
      false).front();
  ASSERT_EQ(*resultEval, LiteralBool(false));

  // if all are true => evaluate to false
  resultEval = ast.evaluateCircuit(
      {{"alpha", new LiteralBool(true)},
       {"beta", new LiteralBool(true)},
       {"gamma", new LiteralBool(true)},
       {"delta", new LiteralBool(true)}},
      false).front();
  ASSERT_EQ(*resultEval, LiteralBool(false));
}

TEST(NodeTest, testUniqueNodeId_Call) {
  // This test makes sure that
  auto callNode = new Call(new Function("computeX"));
  ASSERT_EQ(callNode->AbstractExpr::getUniqueNodeId(), callNode->AbstractStatement::getUniqueNodeId());
}

TEST(NodeTest, testUniqueNodeId_CallExternal__UnfixedTest) {
  // TODO: Find a solution on how to handle double inheritance of CallExternal and associated base class Node object
  auto callExtNode = new CallExternal("genSecretKeys");
  ASSERT_EQ(callExtNode->AbstractExpr::getUniqueNodeId(), callExtNode->AbstractStatement::getUniqueNodeId());
}


