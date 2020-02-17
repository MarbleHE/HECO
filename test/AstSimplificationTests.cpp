#include <regex>
#include "SecretTaintingVisitor.h"
#include "Ast.h"
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"
#include "Function.h"

TEST(SecretTaintingTest, simpleAst_noTaintingForPlaintextComputationsOnly) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(5, ast);

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);

  // check that no element is tainted
  ASSERT_TRUE(stv.getSecretTaintingList().empty());
}

TEST(SecretTaintingTest, simpleAst_singleStatementTainted) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(5, ast);

  // modify the AST's function parameter
  auto f = ast.getRootNode()->castTo<Function>();
  // overwrite the (Types::int inputC) parameter
  f->getParams().back()->setAttributes(
      new Datatype(Types::INT, true),
      new Variable("inputC"));

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);
  // make sure that the visitor collected anything
  ASSERT_FALSE(stv.getSecretTaintingList().empty());

  // find unique node ID of return statement: this and all children of Return should be tainted
  const std::regex varAssignmRegex("VarAssignm_[0-9]+");
  auto allNodes = ast.getAllNodes([&](Node *n) {
    std::cout << n->getUniqueNodeId() << std::endl;
    return std::regex_match(n->getUniqueNodeId(), std::regex("VarAssignm_[0-9]+"));
  });
  auto taintedRoot = *allNodes.begin();
  std::set<std::string> taintedNodeUniqueIds({taintedRoot->getUniqueNodeId()});
  for (auto &node : taintedRoot->getDescendants()) taintedNodeUniqueIds.insert(node->getUniqueNodeId());

  // check tainted status
  EXPECT_EQ(stv.getSecretTaintingList().size(), taintedNodeUniqueIds.size());
  for (auto &nodeId : stv.getSecretTaintingList()) {
    EXPECT_TRUE(
        std::find(taintedNodeUniqueIds.begin(), taintedNodeUniqueIds.end(), nodeId)!=taintedNodeUniqueIds.end());
  }
}

TEST(SecretTaintingTest, simpleAst_multipleStatementsTainted) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(16, ast);

  // modify the AST's function parameter
  auto f = ast.getRootNode()->castTo<Function>();
  // overwrite the (Types::int inputC) parameter
  f->getParams().at(0)->setAttributes(
      new Datatype(Types::INT, true),
      new Variable("inputA"));

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);
  // make sure that the visitor collected anything
  ASSERT_FALSE(stv.getSecretTaintingList().empty());

  // find unique node ID of last three statements
  std::set<std::string> taintedNodeUniqueIds;
  auto addStatementAndItsDescendants = [&taintedNodeUniqueIds](Node *n) {
    taintedNodeUniqueIds.insert(n->getUniqueNodeId());
    for (auto &node : n->getDescendants()) taintedNodeUniqueIds.insert(node->getUniqueNodeId());
  };
  // return prod*3;
  addStatementAndItsDescendants(*f->getBody().end());
  // prod = prod * inputC;
  addStatementAndItsDescendants(*(f->getBody().end() - 1));
  // int prod = inputA * inputB;
  addStatementAndItsDescendants(*(f->getBody().end() - 2));

  // check tainted status
  // check tainted status
  EXPECT_EQ(stv.getSecretTaintingList().size(), taintedNodeUniqueIds.size());
  for (auto &nodeId : stv.getSecretTaintingList()) {
    EXPECT_TRUE(
        std::find(taintedNodeUniqueIds.begin(), taintedNodeUniqueIds.end(), nodeId)!=taintedNodeUniqueIds.end());
  }
}







