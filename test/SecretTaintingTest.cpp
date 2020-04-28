#include <regex>
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/visitor/SecretTaintingVisitor.h"
#include "ast_opt/utilities/DotPrinter.h"
#include "ast_opt/ast/Ast.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/VarAssignm.h"
#include "AstTestingGenerator.h"
#include "gtest/gtest.h"

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
    for (auto funcParam : functionNode.castTo<Function>()->getParameters()) {
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

  //  Function_0: (computePrivate) [global]      // tainted
  //    ParameterList_3: [Function_0]            // tainted
  //        FunctionParameter_6:
  //            Datatype_4: (plaintext int)
  //            Variable_5: (inputA)
  //        FunctionParameter_9:
  //            Datatype_7: (plaintext int)
  //            Variable_8: (inputB)
  //        FunctionParameter_12:                // tainted
  //            Datatype_10: (encrypted int)     // tainted
  //            Variable_11: (inputC)            // tainted
  //    Block_2:                                 // tainted
  //        VarDecl_17: (prod) [Block_2]
  //            Datatype_18: (plaintext int)
  //            ArithmeticExpr_15:
  //                Variable_13: (inputA)
  //                Operator_16: (mult)
  //                Variable_14: (inputB)
  //        VarAssignm_23: (argPow)              // tainted
  //            ArithmeticExpr_21:               // tainted
  //                Variable_19: (inputC)        // tainted
  //                Operator_22: (mult)          // tainted
  //                Variable_20: (inputC)        // tainted
  //        Return_29:
  //            ArithmeticExpr_27:
  //                Variable_24: (prod)
  //                Operator_28: (div)
  //                LiteralInt_26: (3)

  // check that exactly 11 nodes are tainted
  EXPECT_EQ(stv.getSecretTaintingList().size(), 11);

  // Function_0
  auto function = ast.getRootNode()->castTo<Function>();
  expectedTaintedNodeIds.insert(function->getUniqueNodeId());

  // ParameterList_3
  expectedTaintedNodeIds.insert(function->getParameterList()->getUniqueNodeId());

  // FunctionParameter_12 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(ast.getRootNode()->castTo<Function>()->getParameters().back());

  // Block_2
  expectedTaintedNodeIds.insert(function->getBody()->getUniqueNodeId());

  // VarAssignm_23 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getBodyStatements().at(1));

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList())
    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1) << "Node (" << nodeId << ") is not expected to be tainted.";
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
  EXPECT_EQ(stv.getSecretTaintingList().size(), 22);

  // Function_0: (computePrivate) [global]        // tainted
  //     ParameterList_3: [Function_0]            // tainted
  //         FunctionParameter_6:                 // tainted
  //             Datatype_4: (encrypted int)      // tainted
  //             Variable_5: (inputA)             // tainted
  //         FunctionParameter_9:
  //             Datatype_7: (plaintext int)
  //             Variable_8: (inputB)
  //         FunctionParameter_12:
  //             Datatype_10: (plaintext int)
  //             Variable_11: (inputC)
  //     Block_2:                                 // tainted
  //         VarDecl_17: (prod) [Block_2]         // tainted
  //             Datatype_18: (plaintext int)     // tainted
  //             ArithmeticExpr_15:               // tainted
  //                 Variable_13: (inputA)        // tainted
  //                 Operator_16: (mult)          // tainted
  //                 Variable_14: (inputB)        // tainted
  //         VarAssignm_23: (prod)                // tainted
  //             ArithmeticExpr_21:               // tainted
  //                 Variable_19: (prod)          // tainted
  //                 Operator_22: (mult)          // tainted
  //                 Variable_20: (inputC)        // tainted
  //         Return_29:                           // tainted
  //             ArithmeticExpr_27:               // tainted
  //                 Variable_24: (prod)          // tainted
  //                 Operator_28: (mult)          // tainted
  //                 LiteralInt_26: (3)           // tainted

  auto function = ast.getRootNode()->castTo<Function>();
  // Function_0
  expectedTaintedNodeIds.insert(function->getUniqueNodeId());

  // ParameterList_3
  expectedTaintedNodeIds.insert(function->getParameterList()->getUniqueNodeId());

  // FunctionParameter_6 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getParameters().front());

  // Block_2 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getBody());

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList())
    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1) << "Node (" << nodeId << ") is not expected to be tainted.";
}

TEST_F(SecretTaintingFixture, complexAst_multipleNonSequentialStatementsTainted) { /* NOLINT */
  generateAst(22);

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);
  // make sure that the visitor collected anything
  ASSERT_FALSE(stv.getSecretTaintingList().empty());
  EXPECT_EQ(stv.getSecretTaintingList().size(), 40);

  //  Function_0: (computeTotal) [global]                                                      // tainted
  //      ParameterList_1: [Function_0]
  //          FunctionParameter_5:
  //              Datatype_3: (plaintext int)
  //              Variable_4: (subtotal)
  //      Block_2:                                                                             // tainted
  //          VarDecl_14: (qualifiesForSpecialDiscount) [Block_2]
  //              Datatype_6: (plaintext bool)
  //              UnaryExpr_12:
  //                  Operator_13: (!)
  //                  LogicalExpr_10:
  //                      Variable_7: (subtotal)
  //                      Operator_11: (<)
  //                      LiteralInt_9: (1000)
  //          VarDecl_51: (discount)                                                           // tainted
  //              Datatype_45: (encrypted float)                                               // tainted
  //              Call_49:                                                                     // tainted
  //                  FunctionParameter_48: [Block_2]                                          // tainted
  //                      Datatype_46: (encrypted bool)                                        // tainted
  //                      Variable_47: (qualifiesForSpecialDiscount)                           // tainted
  //                  Function_15: (computeDiscountOnServer) [Call_49]                         // tainted
  //                      ParameterList_16: [Function_15]                                      // tainted
  //                          FunctionParameter_20:                                            // tainted
  //                              Datatype_18: (encrypted bool)                                // tainted
  //                              Variable_19: (qualifiesForSpecialDiscount)                   // tainted
  //                      Block_17:                                                            // tainted
  //                          VarDecl_24: (discountRate) [Block_17]                            // tainted
  //                              Datatype_21: (encrypted float)                               // tainted
  //                              LiteralFloat_23: (0)                                         // tainted
  //                          Block_42:                                                        // tainted
  //                              VarAssignm_41: (discountRate) [Block_42]                     // tainted
  //                                  ArithmeticExpr_39:                                       // tainted
  //                                      ArithmeticExpr_28:                                   // tainted
  //                                          Variable_25: (qualifiesForSpecialDiscount)       // tainted
  //                                          Operator_29: (mult)                              // tainted
  //                                          LiteralFloat_27: (0.9)                           // tainted
  //                                      Operator_40: (add)                                   // tainted
  //                                      ArithmeticExpr_37:                                   // tainted
  //                                          ArithmeticExpr_33:                               // tainted
  //                                              LiteralInt_31: (1)                           // tainted
  //                                              Operator_34: (sub)                           // tainted
  //                                              Variable_32: (qualifiesForSpecialDiscount)   // tainted
  //                                          Operator_38: (mult)                              // tainted
  //                                          LiteralFloat_36: (0.98)                          // tainted
  //                          Return_44: [Block_17]                                            // tainted
  //                              Variable_43: (discountRate)                                  // tainted
  //          Return_56:                                                                       // tainted
  //              ArithmeticExpr_54:                                                           // tainted
  //                  Variable_52: (subtotal)                                                  // tainted
  //                  Operator_55: (mult)                                                      // tainted
  //                  Variable_53: (discount)                                                  // tainted

  auto func = ast.getRootNode()->castTo<Function>();

  // Function_0
  expectedTaintedNodeIds.insert(func->getUniqueNodeId());

  // Block_2
  expectedTaintedNodeIds.insert(func->getBody()->getUniqueNodeId());

  // VarDecl_51 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(func->getBodyStatements().at(1));

  // Return_56 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(func->getBodyStatements().at(2));

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList())
    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1)
              << "Node (" << nodeId << ") is tainted but not expected to be tainted.";
}

TEST_F(SecretTaintingFixture, unsupportedStatement_While) {  /* NOLINT */
  // id=11 generates an AST containing an unsupported While statement
  generateAst(11);

  // perform tainting
  SecretTaintingVisitor stv;
  ASSERT_THROW(stv.visit(ast), std::invalid_argument);
}

TEST_F(SecretTaintingFixture, unsupportedStatement_If) {  /* NOLINT */
  generateAst(36);

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);
  EXPECT_EQ(stv.getSecretTaintingList().size(), 9);

  // Function_0: (simpleIfConditionalAssignment) [global]  // tainted
  //     ParameterList_1: [Function_0]                     // tainted
  //         FunctionParameter_5:                          // tainted
  //             Datatype_3: (encrypted int)               // tainted
  //             Variable_4: (cond)                        // tainted
  //     Block_2:                                          // tainted
  //         VarDecl_6: (a) [Block_2]
  //             Datatype_7: (plaintext int)
  //             LiteralInt_9: (1)
  //         If_21:                                         // tainted
  //             LogicalExpr_13:                            // tainted
  //                 Variable_10: (cond)                    // tainted
  //                 Operator_14: (>)
  //                 LiteralInt_12: (11)
  //             VarAssignm_17: (a) [VarAssignm_17]
  //                 LiteralInt_16: (83)
  //             VarAssignm_20: (a) [Block_2]
  //                 LiteralInt_19: (11)
  //         Return_23:
  //             Variable_22: (a)

  auto function = ast.getRootNode()->castTo<Function>();
  // Function_0
  expectedTaintedNodeIds.insert(function->getUniqueNodeId());

  // ParameterList_1 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getParameterList());

  // Block_2
  expectedTaintedNodeIds.insert(function->getBody()->getUniqueNodeId());

  // If_21
  expectedTaintedNodeIds.insert(function->getBody()->getChildAtIndex(1)->getUniqueNodeId());

  // LogicalExpr_13
  expectedTaintedNodeIds.insert(function->getBody()->getChildAtIndex(1)->getChildAtIndex(0)->getUniqueNodeId());

  // Variable_10
  expectedTaintedNodeIds
      .insert(function->getBody()->getChildAtIndex(1)->getChildAtIndex(0)->getChildAtIndex(0)->getUniqueNodeId());

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList())
    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1)
              << "Node (" << nodeId << ") is tainted but not expected to be tainted.";
}

TEST_F(SecretTaintingFixture, unsupportedStatement_CallExternal) {  /* NOLINT */
  // id=3 generates an AST containing an unsupported CallExternal object
  generateAst(3);

  // perform tainting
  SecretTaintingVisitor stv;
  ASSERT_THROW(stv.visit(ast), std::invalid_argument);
}
