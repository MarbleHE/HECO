#include <regex>
#include <PrintVisitor.h>
#include "SecretTaintingVisitor.h"
#include "Ast.h"
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"
#include "Function.h"
#include "Call.h"
#include "Block.h"
#include "VarAssignm.h"

class SecretTaintingFixture : public ::testing::Test {
 protected:
  Ast ast;
  std::set<std::string> expectedTaintedNodeIds;

 public:
  SecretTaintingFixture() = default;

  void generateAst(int demoAstId) {
    AstTestingGenerator::generateAst(demoAstId, ast);
  }

  static void setParamEncrypted(AbstractNode &functionNode, const std::string &variableIdentifier) {
    for (auto funcParam : functionNode.castTo<Function>()->getParams()) {
      auto var = dynamic_cast<Variable *>(funcParam->getValue());
      if (var!=nullptr && var->getIdentifier()==variableIdentifier) {
        funcParam->getDatatype()->setEncrypted(true);
      }
    }
  }

  void addStatementAndItsDescendantsToExpectedTaintedNodes(AbstractNode *n) {
    expectedTaintedNodeIds.insert(n->getUniqueNodeId());
    for (auto &node : n->getDescendants()) {
      expectedTaintedNodeIds.insert(node->getUniqueNodeId());
    }
  }
};

TEST_F(SecretTaintingFixture, simpleAst_noTaintingForPlaintextComputationsOnly) { /* NOLINT */
  // generate an AST without any encrypted parameters
  generateAst(5);

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);

  // check that no element is tainted
  ASSERT_TRUE(stv.getSecretTaintingList().empty());
}

TEST_F(SecretTaintingFixture, simpleAst_singleStatementTainted) { /* NOLINT */
  generateAst(5);

  // modify the AST's function parameter to make inputC encrypted
  setParamEncrypted(*ast.getRootNode(), "inputC");

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);
  // make sure that the visitor collected anything
  ASSERT_FALSE(stv.getSecretTaintingList().empty());

  // find unique node ID of return statement: this and all children of Return should be tainted
  auto taintedRoot = ast.getRootNode()->castTo<Function>()->getBody().at(1);
  addStatementAndItsDescendantsToExpectedTaintedNodes(taintedRoot);
  // also the FunctionParameter and associated Datatype should be tainted (last Parameter 'inputC' = back)
  addStatementAndItsDescendantsToExpectedTaintedNodes(ast.getRootNode()->castTo<Function>()->getParams().back());
  // add Function itself
  expectedTaintedNodeIds.insert(ast.getRootNode()->getUniqueNodeId());

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList()) EXPECT_TRUE(expectedTaintedNodeIds.count(nodeId)==1);
}

TEST_F(SecretTaintingFixture, simpleAst_multipleStatementsTainted) { /* NOLINT */
  generateAst(16);

  // modify the AST's function parameter
  setParamEncrypted(*ast.getRootNode(), "inputA");

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);
  // make sure that the visitor collected anything
  ASSERT_FALSE(stv.getSecretTaintingList().empty());

  // find the unique node IDs of the last three statements - those are the statements expected to be tainted
  auto f = ast.getRootNode()->castTo<Function>();
  // FunctionParameter 'inputA', its associated Datatype and Variable objects
  addStatementAndItsDescendantsToExpectedTaintedNodes(f->getParams().front());
  // int prod = inputA * inputB;
  addStatementAndItsDescendantsToExpectedTaintedNodes(*std::prev(f->getBody().end(), 3));
  // prod = prod * inputC;
  addStatementAndItsDescendantsToExpectedTaintedNodes(*std::prev(f->getBody().end(), 2));
  // return prod*3;
  addStatementAndItsDescendantsToExpectedTaintedNodes(*std::prev(f->getBody().end(), 1));
  // as all statements are tainted, the function is expected to be tainted too
  expectedTaintedNodeIds.insert(f->getUniqueNodeId());

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList()) EXPECT_TRUE(expectedTaintedNodeIds.count(nodeId)==1);
}

TEST_F(SecretTaintingFixture, complexAst_multipleNonSequentialStatementsTainted) { /* NOLINT */
  generateAst(22);

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);
  // make sure that the visitor collected anything
  ASSERT_FALSE(stv.getSecretTaintingList().empty());

  // add the statements expected to be tainted to expectedTaintedNodeIds
  auto func = ast.getRootNode()->castTo<Function>();
  // Function computeTotal
  expectedTaintedNodeIds.insert(func->getUniqueNodeId());
  // VarAssignm secret_float discount = computeDiscountOnServer(...)
  expectedTaintedNodeIds.insert(func->getBody().at(1)->getUniqueNodeId());
  // Datatype of VarAssignm
  expectedTaintedNodeIds.insert(func->getBody().at(1)->castTo<VarDecl>()->getDatatype()->getUniqueNodeId());
  // Call computeDiscountOnServer, it's associated FunctionParameter (parameter values for the called function)
  // and Datatype
  auto callComputeDiscountOnServer = func->getBody().at(1)->castTo<VarDecl>()->getInitializer()->castTo<Call>();
  expectedTaintedNodeIds.insert(callComputeDiscountOnServer->AbstractExpr::getUniqueNodeId());
  addStatementAndItsDescendantsToExpectedTaintedNodes(callComputeDiscountOnServer->getArguments().front());
  // Function computeDiscountOnServer associated to the Call
  auto funcComputeDiscountOnServer = callComputeDiscountOnServer->getFunc();
  expectedTaintedNodeIds.insert(funcComputeDiscountOnServer->getUniqueNodeId());
  // FunctionParameter 'bool qualifiesForSpecialDiscount', its associated Datatype and Variable
  expectedTaintedNodeIds.insert(funcComputeDiscountOnServer->getParams().front()->getUniqueNodeId());
  addStatementAndItsDescendantsToExpectedTaintedNodes(funcComputeDiscountOnServer->getParams().front());
  // VarDecl (statement 0 in computeDiscountOnServer)
  addStatementAndItsDescendantsToExpectedTaintedNodes(funcComputeDiscountOnServer->getBody().at(0));
  // Block (statement 1 in computeDiscountOnServer)
  expectedTaintedNodeIds.insert(funcComputeDiscountOnServer->getBody().at(1)->getUniqueNodeId());
  for (auto statement : *funcComputeDiscountOnServer->getBody().at(1)->castTo<Block>()->getStatements())
    addStatementAndItsDescendantsToExpectedTaintedNodes(statement);
  // all other statements in computeDiscountOnServer body
  auto cdosIterator = funcComputeDiscountOnServer->getBody().begin();
  // advance the iterator by 2 as we already handled statement 0 (VarDecl) and statement 1 (Block)
  std::advance(cdosIterator, 2);
  for (; cdosIterator!=funcComputeDiscountOnServer->getBody().end(); ++cdosIterator) {
    addStatementAndItsDescendantsToExpectedTaintedNodes(*cdosIterator);
  }
  // return subtotal*discount;
  addStatementAndItsDescendantsToExpectedTaintedNodes(func->getBody().at(2));

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList()) EXPECT_TRUE(expectedTaintedNodeIds.count(nodeId)==1);
}

TEST_F(SecretTaintingFixture, unsupportedStatement_While) {  /* NOLINT */
  // id=11 generates an AST containing an unsupported While statement
  generateAst(11);

  // perform tainting
  SecretTaintingVisitor stv;
  ASSERT_THROW(stv.visit(ast), std::invalid_argument);
}

TEST_F(SecretTaintingFixture, unsupportedStatement_If) {  /* NOLINT */
  // id=4 generates an AST containing an unsupported If statement
  generateAst(4);

  // perform tainting
  SecretTaintingVisitor stv;
  ASSERT_THROW(stv.visit(ast), std::invalid_argument);
}

TEST_F(SecretTaintingFixture, unsupportedStatement_CallExternal) {  /* NOLINT */
  // id=3 generates an AST containing an unsupported CallExternal object
  generateAst(3);

  // perform tainting
  SecretTaintingVisitor stv;
  ASSERT_THROW(stv.visit(ast), std::invalid_argument);
}

