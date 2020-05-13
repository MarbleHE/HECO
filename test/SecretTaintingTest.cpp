#include <regex>
#include "ast_opt/visitor/CompileTimeExpressionSimplifier.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/visitor/SecretTaintingVisitor.h"
#include "ast_opt/utilities/DotPrinter.h"
#include "ast_opt/ast/Ast.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/VarAssignm.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/MatrixAssignm.h"
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

  void addNodeToExpectedTaintedNodes(AbstractNode *n) {
    expectedTaintedNodeIds.insert(n->getUniqueNodeId());
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
  //                Operator_22: (mult)
  //                Variable_20: (inputC)        // tainted
  //        Return_29:
  //            ArithmeticExpr_27:
  //                Variable_24: (prod)
  //                Operator_28: (div)
  //                LiteralInt_26: (3)

  // check that exactly 10 nodes are tainted
  EXPECT_EQ(stv.getSecretTaintingList().size(), 10);

  // Function_0
  auto function = ast.getRootNode()->castTo<Function>();
  expectedTaintedNodeIds.insert(function->getUniqueNodeId());

  // ParameterList_3
  expectedTaintedNodeIds.insert(function->getParameterList()->getUniqueNodeId());

  // FunctionParameter_12 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(ast.getRootNode()->castTo<Function>()->getParameters().back());

  // Block_2
  expectedTaintedNodeIds.insert(function->getBody()->getUniqueNodeId());

  // VarAssignm_23, the arithmetic expression and both variables, but not the operator
  auto varAssignm = dynamic_cast<VarAssignm *>(function->getBodyStatements().at(1));
  expectedTaintedNodeIds.insert(varAssignm->getUniqueNodeId());
  auto arithExpr = dynamic_cast<ArithmeticExpr *>(varAssignm->getValue());
  expectedTaintedNodeIds.insert(arithExpr->getUniqueNodeId());
  expectedTaintedNodeIds.insert(arithExpr->getLeft()->getUniqueNodeId()); //lhs
  expectedTaintedNodeIds.insert(arithExpr->getRight()->getUniqueNodeId()); //rhs

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
  //             Datatype_18: (plaintext int)
  //             ArithmeticExpr_15:               // tainted
  //                 Variable_13: (inputA)        // tainted
  //                 Operator_16: (mult)
  //                 Variable_14: (inputB)
  //         VarAssignm_23: (prod)                // tainted
  //             ArithmeticExpr_21:               // tainted
  //                 Variable_19: (prod)          // tainted
  //                 Operator_22: (mult)
  //                 Variable_20: (inputC)
  //         Return_29:                           // tainted
  //             ArithmeticExpr_27:               // tainted
  //                 Variable_24: (prod)          // tainted
  //                 Operator_28: (mult)
  //                 LiteralInt_26: (3)

  auto function = ast.getRootNode()->castTo<Function>();
  // Function_0
  expectedTaintedNodeIds.insert(function->getUniqueNodeId());

  // ParameterList_3
  expectedTaintedNodeIds.insert(function->getParameterList()->getUniqueNodeId());

  // FunctionParameter_6 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getParameters().front());

  // Block_2
  expectedTaintedNodeIds.insert(function->getBody()->getUniqueNodeId());

  // VarDecl_17 + ArithmeticExpr_15 and lhs
  auto varDecl = dynamic_cast<VarDecl *>(function->getBodyStatements()[0]);
  addNodeToExpectedTaintedNodes(varDecl);
  auto arithExpr15 = dynamic_cast<ArithmeticExpr*>(varDecl->getInitializer());
  addNodeToExpectedTaintedNodes(arithExpr15);
  addNodeToExpectedTaintedNodes(arithExpr15->getLeft());

  // VarAssignm_23 + expr + lhs
  auto varAssignm = dynamic_cast<VarAssignm *>(function->getBodyStatements()[1]);
  addNodeToExpectedTaintedNodes(varAssignm);
  auto arithExpr21 = dynamic_cast<ArithmeticExpr*>(varAssignm->getValue());
  addNodeToExpectedTaintedNodes(arithExpr21);
  addNodeToExpectedTaintedNodes(arithExpr21->getLeft());

  // Return_29 + expr + lhs
  auto returnStmt = dynamic_cast<Return*>(function->getBodyStatements()[2]);
  addNodeToExpectedTaintedNodes(returnStmt);
  auto arithExpr27 = dynamic_cast<ArithmeticExpr*>(returnStmt->getReturnExpressions()[0]);
  addNodeToExpectedTaintedNodes(arithExpr27);
  addNodeToExpectedTaintedNodes(arithExpr27->getLeft());


  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList())
    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1) << "Node (" << nodeId << ") is not expected to be tainted.";
  for (auto &nodeId : expectedTaintedNodeIds)
    EXPECT_EQ(stv.getSecretTaintingList().count(nodeId), 1)
              << "Node (" << nodeId << ") is not tainted but expected to be tainted.";
}

TEST_F(SecretTaintingFixture, complexAst_multipleNonSequentialStatementsTainted) { /* NOLINT */
  generateAst(22);

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);
  // make sure that the visitor collected anything
  ASSERT_FALSE(stv.getSecretTaintingList().empty());

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
  //                                          Operator_29: (mult)
  //                                          LiteralFloat_27: (0.9)
  //                                      Operator_40: (add)
  //                                      ArithmeticExpr_37:                                   // tainted
  //                                          ArithmeticExpr_33:                               // tainted
  //                                              LiteralInt_31: (1)
  //                                              Operator_34: (sub)
  //                                              Variable_32: (qualifiesForSpecialDiscount)   // tainted
  //                                          Operator_38: (mult)
  //                                          LiteralFloat_36: (0.98)
  //                          Return_44: [Block_17]                                            // tainted
  //                              Variable_43: (discountRate)                                  // tainted
  //          Return_56:                                                                       // tainted
  //              ArithmeticExpr_54:                                                           // tainted
  //                  Variable_52: (subtotal)
  //                  Operator_55: (mult)
  //                  Variable_53: (discount)                                                  // tainted

  auto func = ast.getRootNode()->castTo<Function>();

  // Function_0
  expectedTaintedNodeIds.insert(func->getUniqueNodeId());

  // Block_2
  expectedTaintedNodeIds.insert(func->getBody()->getUniqueNodeId());

  // VarDecl_51 and its descendants
  addStatementAndItsDescendantsToExpectedTaintedNodes(func->getBodyStatements().at(1));

  // Return_56 , arith expr +  rhs
  addNodeToExpectedTaintedNodes(func->getBodyStatements().at(2));
  auto arithExpr = dynamic_cast<ArithmeticExpr*>(func->getBodyStatements().at(2)->getChildren()[0]);
  addNodeToExpectedTaintedNodes(arithExpr);
  addNodeToExpectedTaintedNodes(arithExpr->getRight());

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList())
    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1)
              << "Node (" << nodeId << ") is tainted but not expected to be tainted.";
  for (auto &nodeId : expectedTaintedNodeIds)
    EXPECT_EQ(stv.getSecretTaintingList().count(nodeId), 1)
              << "Node (" << nodeId << ") is not tainted but expected to be tainted.";
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

TEST_F(SecretTaintingFixture, publicTurnedSecret) { /* NOLINT */
  // id=62 generates AST for:
  // int publicTurnedSecret(int x) { // x secret
  //    int k; // k public
  //    k = x + 5; // k now tainted
  //    return k;
  // }
  generateAst(62);

//  Function_0: (publicTurnedSecret)	[global].......//tainted
//    ParameterList_1:	[Function_0]...................//tainted
//      FunctionParameter_5:...........................//tainted
//        Datatype_3: (encrypted int)..................//tainted
//        Variable_4: (x)..............................//tainted
//  Block_2:...........................................//tainted
//    VarDecl_7: (k)	[Block_2]
//      Datatype_6: (plaintext int)
//    VarAssignm_13: (k)...............................//tainted
//      ArithmeticExpr_11:.............................//tainted
//        Variable_8: (x)..............................//tainted
//        Operator_12: (add)
//        LiteralInt_10: (5)
//    Return_15:.......................................//tainted
//      Variable_14: (k)...............................//tainted

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);

  expectedTaintedNodeIds.clear();
  //function should be tainted
  auto function = dynamic_cast<Function *>(ast.getRootNode());
  expectedTaintedNodeIds.insert(function->getUniqueNodeId());
  // parameter list and children should be tainted
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getParameterList());
  // function body should be tainted
  expectedTaintedNodeIds.insert(function->getBody()->getUniqueNodeId());
  // second statement in function body should be tainted
  auto varAssignm = dynamic_cast<VarAssignm *>(function->getBodyStatements()[1]);
  expectedTaintedNodeIds.insert(varAssignm->getUniqueNodeId());
  // binary expression should be tainted
  expectedTaintedNodeIds.insert(varAssignm->getValue()->getUniqueNodeId());
  // lhs of binary expression should be tainted
  expectedTaintedNodeIds.insert(varAssignm->getValue()->getChildren()[0]->getUniqueNodeId());
  // third stmt (return) in function body and its subnodes should be tainted
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getBodyStatements()[2]);

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList())
    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1)
              << "Node (" << nodeId << ") is tainted but not expected to be tainted.";
  for (auto &nodeId : expectedTaintedNodeIds)
    EXPECT_EQ(stv.getSecretTaintingList().count(nodeId), 1)
              << "Node (" << nodeId << ") is not tainted but expected to be tainted.";
}

TEST_F(SecretTaintingFixture, publicTurnedSecretMatrix) { /* NOLINT */
  // id=63 generates AST for:
  // int publicTurnedSecret(int x) { // x secret
  //    int k; // k public
  //    k[0][0] = 7; // still public
  //    k[0][1] = x + 5; // k now tainted
  //    return k;
  // }
  generateAst(63);

  //  Function_0: (publicTurnedSecret)	[global].......//tainted
  //	ParameterList_1:	[Function_0]...............//tainted
  //		FunctionParameter_5:.......................//tainted
  //			Datatype_3: (encrypted int)............//tainted
  //			Variable_4: (x)........................//tainted
  //	Block_2:
  //		VarDecl_7: (k)	[Block_2]
  //			Datatype_6: (plaintext int)
  //		MatrixAssignm_16:
  //(0,0)		MatrixElementRef_13:
  //				Variable_8: (k)
  //				LiteralInt_10: (0)
  //				LiteralInt_12: (0)
  //			LiteralInt_15: (7)
  //		MatrixAssignm_28:...........................//tainted
  //(0,0)		MatrixElementRef_22:
  //				Variable_17: (k)
  //				LiteralInt_19: (0)
  //				LiteralInt_21: (1)
  //			ArithmeticExpr_26:......................//tainted
  //				Variable_23: (x)....................//tainted
  //				Operator_27: (add)
  //				LiteralInt_25: (5)
  //		Return_30:..................................//tainted
  //			Variable_29: (k)........................//tainted

  // perform tainting
  SecretTaintingVisitor stv;
  stv.visit(ast);

  expectedTaintedNodeIds.clear();
  //function should be tainted
  auto function = dynamic_cast<Function *>(ast.getRootNode());
  addNodeToExpectedTaintedNodes(function);
  // parameter list and children should be tainted
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getParameterList());
  // function body should be tainted
  addNodeToExpectedTaintedNodes(function->getBody());
  // second MatrixAssignm should be tainted
  auto matrixAssignm = dynamic_cast<MatrixAssignm *>(function->getBodyStatements()[2]);
  addNodeToExpectedTaintedNodes(matrixAssignm);
  // binary expression should be tainted
  addNodeToExpectedTaintedNodes(matrixAssignm->getValue());
  // lhs of binary expression should be tainted
  addNodeToExpectedTaintedNodes(matrixAssignm->getValue()->getChildren()[0]);
  // fourth stmt (return) in function body and its subnodes should be tainted
  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getBodyStatements()[3]);

  // check tainted status
  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
  // - check for each tainted if its expected to be tainted
  for (auto &nodeId : stv.getSecretTaintingList())
    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1)
              << "Node (" << nodeId << ") is tainted but not expected to be tainted.";
  for (auto &nodeId : expectedTaintedNodeIds)
    EXPECT_EQ(stv.getSecretTaintingList().count(nodeId), 1)
              << "Node (" << nodeId << ") is not tainted but expected to be tainted.";
}


// TODO: This test is borked, because whenever CTES changes what it does, it breaks
// TODO: Instead, let's output the AST in a specific state, save that, and use it for testing
//TEST_F(SecretTaintingFixture, taintedMatrixAssignmTest) { /* NOLINT */
//  generateAst(60);
//
//  // perform compile-time expression simplification (limit max unrolled loops to 2)
//  CompileTimeExpressionSimplifier ctes(CtesConfiguration(2));
//  ctes.visit(ast);
//
//  // perform tainting
//  SecretTaintingVisitor stv;
//  stv.visit(ast);
//  EXPECT_EQ(stv.getSecretTaintingList().size(), 30);
//
////  Function_0: (runLaplacianSharpeningAlgorithm)	[global]
////  	ParameterList_1:	[Function_0]...........................................................// tainted
////  		FunctionParameter_5:...................................................................// tainted
////  			Datatype_3: (encrypted int)........................................................// tainted
////  			Variable_4: (img)....................... ..........................................// tainted
////  		FunctionParameter_8:
////  			Datatype_6: (plaintext int)
////  			Variable_7: (imgSize)
////  	Block_2:
////  		VarDecl_3338: (img2)	[Block_2]
////  			Datatype_3339: (plaintext int)
////  		For_157:
////  			VarDecl_139: (x)
////  				Datatype_140: (plaintext int)
////  				LiteralInt_142: (1)
////  			LogicalExpr_149:
////  				Variable_143: (x)
////  				Operator_150: (<)
////  				ArithmeticExpr_145:
////  					Variable_144: (imgSize)
////  					Operator_146: (sub)
////  					LiteralInt_148: (1)
////  			VarAssignm_156: (x)
////  				ArithmeticExpr_154:
////  					Variable_151: (x)
////  					Operator_155: (add)
////  					LiteralInt_153: (1)
////  			Block_138:	[Block_138]
////  				For_137:	[Block_138]
////  					VarDecl_119: (y)
////  						Datatype_120: (plaintext int)
////  						LiteralInt_122: (1)
////  					LogicalExpr_129:
////  						Variable_123: (y)
////  						Operator_130: (<)
////  						ArithmeticExpr_125:
////  							Variable_124: (imgSize)
////  							Operator_126: (sub)
////  							LiteralInt_128: (1)
////  					VarAssignm_136: (y)
////  						ArithmeticExpr_134:
////  							Variable_131: (y)
////  							Operator_135: (add)
////  							LiteralInt_133: (1)
////  					Block_118:	[Block_118]....................................................// tainted
////  						MatrixAssignm_117:	[Block_118]........................................// tainted
////  (0,0)						MatrixElementRef_98:
////  								Variable_88: (img2)
////  								LiteralInt_90: (0)
////  								OperatorExpr_3161:
////  									Operator_97: (add)
////  									OperatorExpr_3162:
////  										Operator_94: (mult)
////  										Variable_91: (imgSize)
////  										Variable_92: (x)
////  									Variable_95: (y)
////  							OperatorExpr_3163:.................................................// tainted
////  								Operator_116: (sub)
////  (0,0)							MatrixElementRef_109:....................... ..................// tainted
////  									Variable_99: (img).........................................// tainted
////  									LiteralInt_101: (0)
////  									OperatorExpr_3164:
////  										Operator_108: (add)
////  										OperatorExpr_3165:
////  											Operator_105: (mult)
////  											Variable_102: (imgSize)
////  											Variable_103: (x)
////  										Variable_106: (y)
////  								OperatorExpr_3166:.............................................// tainted
////  									Operator_114: (div)
////  									OperatorExpr_3337:.........................................// tainted
////  										Operator_3167: (add)
////  (0,0)									MatrixElementRef_3188:.................................// tainted
////  											Variable_3168: (img)...............................// tainted
////  											LiteralInt_3170: (0)
////  											OperatorExpr_3185:
////  												Operator_3171: (add)
////  												OperatorExpr_3181:
////  													Operator_3172: (mult)
////  													Variable_3173: (imgSize)
////  													OperatorExpr_3178:
////  														Operator_3174: (add)
////  														Variable_3175: (x)
////  														LiteralInt_3177: (-1)
////  												Variable_3182: (y)
////  												LiteralInt_3184: (-1)
////  (0,0)									MatrixElementRef_3203:.................................// tainted
////  											Variable_3189: (img)...............................// tainted
////  											LiteralInt_3191: (0)
////  											OperatorExpr_3200:
////  												Operator_3192: (add)
////  												OperatorExpr_3196:
////  													Operator_3193: (mult)
////  													Variable_3194: (imgSize)
////  													Variable_3195: (x)
////  												Variable_3197: (y)
////  												LiteralInt_3199: (-1)
////  (0,0)									MatrixElementRef_3224:.................................// tainted
////  											Variable_3204: (img)...............................// tainted
////  											LiteralInt_3206: (0)
////  											OperatorExpr_3221:
////  												Operator_3207: (add)
////  												OperatorExpr_3217:
////  													Operator_3208: (mult)
////  													Variable_3209: (imgSize)
////  													OperatorExpr_3214:
////  														Operator_3210: (add)
////  														Variable_3211: (x)
////  														LiteralInt_3213: (1)
////  												Variable_3218: (y)
////  												LiteralInt_3220: (-1)
////  (0,0)									MatrixElementRef_3241:.................................// tainted
////  											Variable_3225: (img)...............................// tainted
////  											LiteralInt_3227: (0)
////  											OperatorExpr_3240:
////  												Operator_3228: (add)
////  												OperatorExpr_3238:
////  													Operator_3229: (mult)
////  													Variable_3230: (imgSize)
////  													OperatorExpr_3235:
////  														Operator_3231: (add)
////  														Variable_3232: (x)
////  														LiteralInt_3234: (-1)
////  												Variable_3239: (y)
////  										OperatorExpr_3256:.....................................// tainted
////  											Operator_3242: (mult)
////  (0,0)										MatrixElementRef_3253:.............................// tainted
////  												Variable_3243: (img)...........................// tainted
////  												LiteralInt_3245: (0)
////  												OperatorExpr_3252:
////  													Operator_3246: (add)
////  													OperatorExpr_3250:
////  														Operator_3247: (mult)
////  														Variable_3248: (imgSize)
////  														Variable_3249: (x)
////  													Variable_3251: (y)
////  											LiteralInt_3255: (-8)
////  (0,0)									MatrixElementRef_3279:.................................// tainted
////  											Variable_3263: (img)...............................// tainted
////  											LiteralInt_3265: (0)
////  											OperatorExpr_3278:
////  												Operator_3266: (add)
////  												OperatorExpr_3276:
////  													Operator_3267: (mult)
////  													Variable_3268: (imgSize)
////  													OperatorExpr_3273:
////  														Operator_3269: (add)
////  														Variable_3270: (x)
////  														LiteralInt_3272: (1)
////  												Variable_3277: (y)
////  (0,0)									MatrixElementRef_3300:.................................// tainted
////  											Variable_3280: (img)...............................// tainted
////  											LiteralInt_3282: (0)
////  											OperatorExpr_3297:
////  												Operator_3283: (add)
////  												OperatorExpr_3293:
////  													Operator_3284: (mult)
////  													Variable_3285: (imgSize)
////  													OperatorExpr_3290:
////  														Operator_3286: (add)
////  														Variable_3287: (x)
////  														LiteralInt_3289: (-1)
////  												Variable_3294: (y)
////  												LiteralInt_3296: (1)
////  (0,0)									MatrixElementRef_3315:.................................// tainted
////  											Variable_3301: (img)...............................// tainted
////  											LiteralInt_3303: (0)
////  											OperatorExpr_3312:
////  												Operator_3304: (add)
////  												OperatorExpr_3308:
////  													Operator_3305: (mult)
////  													Variable_3306: (imgSize)
////  													Variable_3307: (x)
////  												Variable_3309: (y)
////  												LiteralInt_3311: (1)
////  (0,0)									MatrixElementRef_3336:.................................// tainted
////  											Variable_3316: (img)...............................// tainted
////  											LiteralInt_3318: (0)
////  											OperatorExpr_3333:
////  												Operator_3319: (add)
////  												OperatorExpr_3329:
////  													Operator_3320: (mult)
////  													Variable_3321: (imgSize)
////  													OperatorExpr_3326:
////  														Operator_3322: (add)
////  														Variable_3323: (x)
////  														LiteralInt_3325: (1)
////  												Variable_3330: (y)
////  												LiteralInt_3332: (1)
////  									LiteralInt_112: (2)
////  		Return_159:	[Block_2]
////  			Variable_158: (img2)
////
//
//  auto function = ast.getRootNode()->castTo<Function>();
//  // ParameterList_1 and the first child (FunctionParameter_5) including its descendants
//  expectedTaintedNodeIds.insert(function->getParameterList()->getUniqueNodeId());
//  addStatementAndItsDescendantsToExpectedTaintedNodes(function->getParameterList()->getChildAtIndex(0));
//
//  // Block_2 -> 2nd statement (For_157) -> For-loop body (Block_138) -> For-loop (For_137)
//  auto outerForLoop = function->getBodyStatements().at(1)->castTo<For>();
//  auto innerForLoop = outerForLoop->getStatementToBeExecuted()->castTo<Block>()->getStatements().at(0)->castTo<For>();
//
//  // add tainted: For-loop (For_137) -> For-loop body (Block_118)
//  auto innerForLoopBlock = innerForLoop->getStatementToBeExecuted();
//  expectedTaintedNodeIds.insert(innerForLoopBlock->getUniqueNodeId());
//
//  // add tainted: 1st statement (MatrixAssignm_117)
//  auto matrixAssignm = innerForLoopBlock->getChildAtIndex(0)->castTo<MatrixAssignm>();
//  expectedTaintedNodeIds.insert(matrixAssignm->getUniqueNodeId());
//
//  // add tainted: all MatrixElementRef
//  // add tainted: all Variable with identifier 'img'
//  for (auto node : matrixAssignm->getValue()->getDescendants()) {
//    auto mxElementRef = dynamic_cast<MatrixElementRef *>(node);
//    auto var = dynamic_cast<Variable *>(node);
//    auto operatorExpr = dynamic_cast<OperatorExpr *>(node);
//    if (mxElementRef!=nullptr
//        || (var!=nullptr && var->getIdentifier()=="img")
//        || (operatorExpr!=nullptr
//            && ((operatorExpr->getOperator()->equals(MULTIPLICATION)
//                && operatorExpr->getOperands().at(1)->isEqual(new LiteralInt(-8)))
//                || (operatorExpr->getOperator()->equals(ADDITION) && operatorExpr->countChildrenNonNull()==10)
//                || (operatorExpr->getOperator()->equals(DIVISION))))) {
//      expectedTaintedNodeIds.insert(node->getUniqueNodeId());
//    }
//  }
//  // add tainted: OperatorExpr with subtraction operator
//  expectedTaintedNodeIds.insert(matrixAssignm->getValue()->getUniqueNodeId());
//
//  // check tainted status
//  // - make sure that the number of expected tainted nodes and the actual number of tainted nodes equals
//  EXPECT_EQ(stv.getSecretTaintingList().size(), expectedTaintedNodeIds.size());
//  // - check for each tainted if its expected to be tainted
//  for (auto &nodeId : stv.getSecretTaintingList())
//    EXPECT_EQ(expectedTaintedNodeIds.count(nodeId), 1)
//              << "Node (" << nodeId << ") is tainted but not expected to be tainted.";
//}
