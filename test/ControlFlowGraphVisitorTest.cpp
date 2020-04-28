#include "AstTestingGenerator.h"
#include "ast_opt/visitor/ControlFlowGraphVisitor.h"
#include "ast_opt/utilities/GraphNode.h"
#include "gtest/gtest.h"

class FlowGraphVisitorFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // This is required because some methods in ControlFlowGraphVisitor rely on the unique node ID which differs
    // depending on the order in which the tests are executed, therefore we need to reset the ongoing ID counter before
    // each test run.
    AbstractNode::resetNodeIdCounter();
  }

  /// A test helper that checks the variables accessed by a GraphNode. Compares the expected variable reads
  /// (expectedReads) and the expected variable writes (expectedWrites) with those that were recorded in gNode.
  /// Returns False if there is a mismatch (read instead of write, or vice versa), if there is an unexpected read/write,
  /// or if there is an expected read/write that did not actually happen.
  /// \param expectedReads The variables expected to have been read by the given gNode (i.e., program statement).
  /// \param expectedWrites The variables expected to have been written by the given gNode (i.e., program statement).
  /// \param gNode The node whose variable accessed should be checked.
  /// \return True if the actual variable reads and writes equal the expected ones.
  static bool checkVariableReadWrites(std::initializer_list<std::string> expectedReads,
                                      std::initializer_list<std::string> expectedWrites, GraphNode *gNode) {
    // generate sets out of given read/write variables to have fast O(1) access
    std::set<std::string> reads(expectedReads.begin(), expectedReads.end());
    std::set<std::string> writes(expectedWrites.begin(), expectedWrites.end());
    // Iterate over all accessed variables and depending on the access type, check whether it is expected to be read or
    // write, delete the variable afterwards from the reads/writes set as gNode's accessedVariables is a set too.
    // If any variable were accessed unexpectedly, abort by returning False.
    for (auto &[varIdentifier, accessType] : gNode->getAccessedVariables()) {
      if (accessType==AccessType::READ && reads.count(varIdentifier)==1) {
        reads.erase(varIdentifier);
      } else if (accessType==AccessType::WRITE && writes.count(varIdentifier)==1) {
        writes.erase(varIdentifier);
      } else {
        return false;  // abort if there is an unexpected read or write variable access
      }
    }
    // If after processing the for-loop there are no expected reads/writes left that did not actually happen, return True.
    return reads.empty() && writes.empty();
  }

  /// Retrieves all nodes that are reachable from the given control flow graph's root node. From this set of nodes,
  /// extracts and returns all of the nodes that have an edge in the dataflow graph.
  /// Note: Nodes that access a variable whose values was not written before, are not included into the data flow graph.
  ///       This, however, should not happen at all because in that case the program/AST is malformed.
  /// \param cfgRootNode The root node of the control flow graph.
  /// \return The set of GraphNode pointers that have an edge in the data flow graph.
  static std::set<GraphNode *> getGraphNodesWithDataflowEdges(GraphNode *cfgRootNode) {
    std::set<GraphNode *> graphNodesWithEdgesInDfg;
    // iterate over all edges in the control flow graph
    for (auto &gNode : cfgRootNode->getControlFlowGraph()->getAllReachableNodes()) {
      if (!gNode->getDataFlowGraph()->getChildren().empty() || !gNode->getDataFlowGraph()->getParents().empty()) {
        graphNodesWithEdgesInDfg.insert(gNode);
      }
    }
    return graphNodesWithEdgesInDfg;
  }

  static bool checkDataFlowGraphEdge(GraphNode *edgeSrc, GraphNode *edgeDst) {
    // check that the edge was added bilateral
    return edgeSrc->getDataFlowGraph()->hasChild(edgeDst) && edgeDst->getDataFlowGraph()->hasParent(edgeSrc);
  }

  static int getDataFlowGraphNumChildren(GraphNode *gNode) {
    return gNode->getDataFlowGraph()->getChildren().size();
  }

  static int getDataFlowGraphNumParents(GraphNode *gNode) {
    return gNode->getDataFlowGraph()->getParents().size();
  }
};

// Expected control flow graph:
//
//              Function_0
//                  │
//                  ▼
//             ParameterList_3
//                  │
//                  ▼
//              Block_2
//                  │
//                  ▼
//               If_35
//                  │
//        ┌─────────┴────────┐
//        ▼                  ▼
//    Block_34           Block_27
//        │                  │
//        ▼                  ▼
//   VarAssignm_33      VarAssignm_26
//        └─────────┬────────┘
//                  ▼
//               If_45
//                  │
//                  ▼
//            VarAssignm_44
//                  │
//                  ▼
//              Return_50
//
TEST_F(FlowGraphVisitorFixture, controlFlowGraphIncludingIfStatement) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  const auto relType = RelationshipType::CTRL_FLOW_GRAPH;
  auto function0 = new GraphNode();
  auto parameterList3 = new GraphNode(relType, {function0});
  auto block2 = new GraphNode(relType, {parameterList3});
  auto if35 = new GraphNode(relType, {block2});
  auto logicalExpr19 = new GraphNode(relType, {if35});
  auto block34 = new GraphNode(relType, {logicalExpr19});
  auto varAssignm33 = new GraphNode(relType, {block34});
  auto block27 = new GraphNode(relType, {logicalExpr19});
  auto varAssignm26 = new GraphNode(relType, {block27});
  auto if45 = new GraphNode(relType, {varAssignm26, varAssignm33});
  auto logicalExpr39 = new GraphNode(relType, {if45});
  auto varAssignm44 = new GraphNode(relType, {logicalExpr39});
  auto return50 = new GraphNode(relType, {varAssignm44, logicalExpr39});   /* NOLINT ignore_unused */

  // check that the CFG's structure is correct
  EXPECT_TRUE(function0->getControlFlowGraph()->areEqualGraphs(fgv.getRootNodeCfg()));

  // check that all nodes of the CFG have a refToOriginalNode set
  auto nodeHasRefToOriginalNodeSet = [](GraphNode *gNode) -> bool { return gNode->getRefToOriginalNode()!=nullptr; };
  auto allNodes = fgv.getRootNodeCfg()->getControlFlowGraph()->getAllReachableNodes();
  EXPECT_TRUE(std::all_of(allNodes.begin(), allNodes.end(), nodeHasRefToOriginalNodeSet));
}

TEST_F(FlowGraphVisitorFixture, traverseAndPrintNodeTest) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  std::stringstream generatedOutput;
  fgv.getRootNodeCfg()->getControlFlowGraph()->traverseAndPrintNodes(generatedOutput);

  std::string expectedOutput = "(1) Function_0\n"
                               "(1) \tParameterList_3\n"
                               "(1) \t\tBlock_2\n"
                               "(1) \t\t\tIf_35\n"
                               "(2) \t\t\t\tLogicalExpr_19\n"
                               "(1) \t\t\t\t\tBlock_27\n"
                               "(1) \t\t\t\t\t\tVarAssignm_26\n"
                               "(1) \t\t\t\t\t\t\tIf_45\n"
                               "(2) \t\t\t\t\t\t\t\tLogicalExpr_39\n"
                               "(1) \t\t\t\t\t\t\t\t\tVarAssignm_44\n"
                               "(0) \t\t\t\t\t\t\t\t\t\tReturn_50\n"
                               "(0) \t\t\t\t\t\t\t\t\tReturn_50\n"
                               "(1) \t\t\t\t\tBlock_34\n"
                               "(1) \t\t\t\t\t\tVarAssignm_33\n"
                               "(1) \t\t\t\t\t\t\tIf_45\n"
                               "    \t\t\t\t\t\t\t... see above, visiting an already visited node ...\n";

  // check that generated output equals the expected one
  EXPECT_EQ(expectedOutput, generatedOutput.str());
}

TEST_F(FlowGraphVisitorFixture, controlFlowGraphIncludingWhileStatement) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(11, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  const auto relType = RelationshipType::CTRL_FLOW_GRAPH;
  auto function0 = new GraphNode();
  auto parameterList3 = new GraphNode(relType, {function0});
  auto block2 = new GraphNode(relType, {parameterList3});
  auto varDecl17 = new GraphNode(relType, {block2});
  auto varDecl19 = new GraphNode(relType, {varDecl17});
  auto while49 = new GraphNode(relType, {varDecl19});
  auto logicalExpr35 = new GraphNode(relType, {while49});
  auto block48 = new GraphNode(relType, {logicalExpr35});
  auto varAssignm41 = new GraphNode(relType, {block48});
  auto varAssignm47 = new GraphNode(relType, {varAssignm41});
  logicalExpr35->getControlFlowGraph()->addParent(varAssignm47);
  auto return51 = new GraphNode(relType, {logicalExpr35});

  // check that the CFG's structure is correct
  EXPECT_TRUE(function0->getControlFlowGraph()->areEqualGraphs(fgv.getRootNodeCfg()));

  // check that all nodes of the CFG have a refToOriginalNode set
  auto nodeHasRefToOriginalNodeSet = [](GraphNode *gNode) -> bool { return gNode->getRefToOriginalNode()!=nullptr; };
  auto allNodes = fgv.getRootNodeCfg()->getControlFlowGraph()->getAllReachableNodes();
  EXPECT_TRUE(std::all_of(allNodes.begin(), allNodes.end(), nodeHasRefToOriginalNodeSet));
}

// Commented output of traverseAndPrintNodes:
//
// (1)  Function_0 (sumNTimes2)
// (1) 	   ParameterList_3 (int inputA)
// (1)        Block_2 ({ ... })
// (1)           VarDecl_9 (int sum = 0;)
// (1)              VarDecl_13 (int base = 2;)
// (1)                 For_38 (for ...)
// (1)                    VarDecl_17 (int i = 0; [For initializer])
// (2)                       LogicalExpr_21 (i <= inputA; [For condition])
// (1)                          Block_37 ({ ... }, [For body])
// (1)                             VarAssignm_36 (sum = sum + base * i; [For body statement 1])
// (2)                                VarAssignm_28 (i=i+1; [For update statement])
// (1)                                   LogicalExpr_21 (i <= 21, [For condition])
//                                       ... see above, visiting an already visited node ...
// (0)                          Return_40
TEST_F(FlowGraphVisitorFixture, controlFlowGraphIncludingForStatement) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(23, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  const auto relType = RelationshipType::CTRL_FLOW_GRAPH;
  auto function0 = new GraphNode();
  auto parameterList3 = new GraphNode(relType, {function0});
  auto block2 = new GraphNode(relType, {parameterList3});
  auto varDecl9 = new GraphNode(relType, {block2});
  auto varDecl13 = new GraphNode(relType, {varDecl9});
  auto for38 = new GraphNode(relType, {varDecl13});
  auto varDecl17 = new GraphNode(relType, {for38});
  auto logicalExpr21 = new GraphNode(relType, {varDecl17});
  auto block37 = new GraphNode(relType, {logicalExpr21});
  auto varAssignm36 = new GraphNode(relType, {block37});
  auto varAssignm28 = new GraphNode(relType, {varAssignm36});
  logicalExpr21->getControlFlowGraph()->addParent(varAssignm28);
  auto return40 = new GraphNode(relType, {logicalExpr21});  /* NOLINT ignore_unused */

  // check that the CFG's structure is correct
  EXPECT_TRUE(function0->getControlFlowGraph()->areEqualGraphs(fgv.getRootNodeCfg()));

  // check that all nodes of the CFG have a refToOriginalNode set
  auto nodeHasRefToOriginalNodeSet = [](GraphNode *gNode) -> bool { return gNode->getRefToOriginalNode()!=nullptr; };
  auto allNodes = fgv.getRootNodeCfg()->getControlFlowGraph()->getAllReachableNodes();
  EXPECT_TRUE(std::all_of(allNodes.begin(), allNodes.end(), nodeHasRefToOriginalNodeSet));
}

TEST_F(FlowGraphVisitorFixture, dataflowGraph_detectedVariableReadWrites) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(23, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  // [Function] sumNTimes2
  auto cfgRootNode = fgv.getRootNodeCfg();
  EXPECT_TRUE(checkVariableReadWrites({}, {}, cfgRootNode));

  // [ParameterList] (int inputA)
  auto parameterList = cfgRootNode->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkVariableReadWrites({}, {"inputA"}, parameterList));

  // [VarDecl] int sum = 0;
  auto varDeclSum = parameterList->getControlFlowGraph()->getOnlyChild()->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkVariableReadWrites({}, {"sum"}, varDeclSum));

  // [VarDecl] int base = 2;
  auto varDeclBase = varDeclSum->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkVariableReadWrites({}, {"base"}, varDeclBase));

  // [For] for (initializer; condition; updateStatement)
  auto forStmt = varDeclBase->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkVariableReadWrites({}, {}, forStmt));

  // [VarDecl] int i = 0;
  auto varDeclI = forStmt->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkVariableReadWrites({}, {"i"}, varDeclI));

  // [LogicalExpr] i <= inputA
  auto logicalExpr = varDeclI->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkVariableReadWrites({"i", "inputA"}, {}, logicalExpr));

  // [Return] return sum;
  auto returnStmt = logicalExpr->getControlFlowGraph()->getChildAtIndex(1);
  EXPECT_TRUE(checkVariableReadWrites({"sum"}, {}, returnStmt));

  // [VarAssignm] sum = logicalExpr + base * i;
  auto varAssignmSum = logicalExpr->getControlFlowGraph()->getChildAtIndex(0)->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkVariableReadWrites({"sum", "base", "i"}, {"sum"}, varAssignmSum));

  // [VarAssignm] i= i + 1;
  auto varAssignmI = varAssignmSum->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkVariableReadWrites({"i"}, {"i"}, varAssignmI));
}

TEST_F(FlowGraphVisitorFixture, dataflowGraph_addedEdgesInBranchingProgram) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);
  fgv.buildDataFlowGraph();

  // bool computeLogical(int inputA, bool strong, bool negate, int inputB)
  //    if (strong == true) {
  //       inputA = inputA * 42;
  //    } else {
  //       inputA = inputA + 42;
  //    }
  //    if (negate) {
  //       inputA = -inputA;
  //    }
  //    return inputA >= inputB
  // }

  // [Function] computeLogical
  auto function0 = fgv.getRootNodeCfg();
  EXPECT_EQ(getDataFlowGraphNumParents(function0), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(function0), 0);
  // [ParameterList] (int inputA, bool strong, bool negate, int inputB)
  auto parameterList3 = function0->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(parameterList3), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(parameterList3), 2 + 1 + 1 + 1);
  // [Block]
  auto block2 = parameterList3->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(block2), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(block2), 0);
  // [If] if (...)
  auto if35 = block2->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(if35), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(if35), 0);
  // [LogicalExpr] (strong == true)
  auto logicalExpr19 = if35->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(logicalExpr19), 1);
  EXPECT_EQ(getDataFlowGraphNumChildren(logicalExpr19), 0);
  // [Block] then branch 1
  auto block27 = logicalExpr19->getControlFlowGraph()->getChildAtIndex(0);
  EXPECT_EQ(getDataFlowGraphNumParents(block27), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(block27), 0);
  // [VarAssignm] inputA = inputA * 42;
  auto varAssignm26 = block27->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(varAssignm26), 1);
  EXPECT_EQ(getDataFlowGraphNumChildren(varAssignm26), 2);
  // [If] if (...)
  auto if45 = varAssignm26->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(if45), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(if45), 0);
  // [LogicalExpr] (negate)
  auto logicalExpr39 = if45->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(logicalExpr39), 1);
  EXPECT_EQ(getDataFlowGraphNumChildren(logicalExpr39), 0);
  // [VarAssignm] inputA = -inputA; // in then branch 2
  auto varAssignm44 = logicalExpr39->getControlFlowGraph()->getChildAtIndex(0);
  EXPECT_EQ(getDataFlowGraphNumParents(varAssignm44), 2);
  EXPECT_EQ(getDataFlowGraphNumChildren(varAssignm44), 1);
  // [Return] inputA >= inputB
  auto return50 = logicalExpr39->getControlFlowGraph()->getChildAtIndex(1);
  EXPECT_EQ(getDataFlowGraphNumParents(return50), 3 + 1);
  EXPECT_EQ(getDataFlowGraphNumChildren(return50), 0);
  // [Block] else branch 1
  auto block34 = logicalExpr19->getControlFlowGraph()->getChildAtIndex(1);
  EXPECT_EQ(getDataFlowGraphNumParents(block34), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(block34), 0);
  // [VarAssignm] inputA = inputA + 42;
  auto varAssignm33 = block34->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(varAssignm33), 1);
  EXPECT_EQ(getDataFlowGraphNumChildren(varAssignm33), 2);

  // inputA
  EXPECT_TRUE(checkDataFlowGraphEdge(parameterList3, varAssignm26));
  EXPECT_TRUE(checkDataFlowGraphEdge(parameterList3, varAssignm33));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignm26, varAssignm44));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignm33, varAssignm44));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignm26, return50));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignm33, return50));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignm44, return50));

  // strong
  EXPECT_TRUE(checkDataFlowGraphEdge(parameterList3, logicalExpr19));

  // negate
  EXPECT_TRUE(checkDataFlowGraphEdge(parameterList3, logicalExpr39));

  // inputB
  EXPECT_TRUE(checkDataFlowGraphEdge(parameterList3, return50));
}

TEST_F(FlowGraphVisitorFixture, dataflowGraph_addedEdgesInForLoop) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(23, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);
  fgv.buildDataFlowGraph();

  // [Function] sumNTimes2
  auto cfgRootNode = fgv.getRootNodeCfg();
  EXPECT_EQ(getDataFlowGraphNumParents(cfgRootNode), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(cfgRootNode), 0);
  // [ParameterList] (int inputA)
  auto parameterList = cfgRootNode->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(parameterList), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(parameterList), 1);
  // [VarDecl] int sum = 0;
  auto varDeclSum = parameterList->getControlFlowGraph()->getOnlyChild()->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(varDeclSum), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(varDeclSum), 2);
  // [VarDecl] int base = 2;
  auto varDeclBase = varDeclSum->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(varDeclBase), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(varDeclBase), 1);
  // [For] for (initializer; condition; updateStatement)
  auto forStmt = varDeclBase->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(forStmt), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(forStmt), 0);
  // [VarDecl] int i = 0;  (For initializer)
  auto varDeclI = forStmt->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(varDeclI), 0);
  EXPECT_EQ(getDataFlowGraphNumChildren(varDeclI), 3);
  // [LogicalExpr] i <= inputA  (For condition)
  auto logicalExpr = varDeclI->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(logicalExpr), 3);
  EXPECT_EQ(getDataFlowGraphNumChildren(logicalExpr), 0);
  // [Return] return sum;
  auto returnStmt = logicalExpr->getControlFlowGraph()->getChildAtIndex(1);
  EXPECT_EQ(getDataFlowGraphNumParents(returnStmt), 2);
  EXPECT_EQ(getDataFlowGraphNumChildren(returnStmt), 0);
  // [VarAssignm] sum = sum + base * i;
  auto varAssignmSum = logicalExpr->getControlFlowGraph()->getChildAtIndex(0)->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(varAssignmSum), 2 + 1 + 2);
  EXPECT_EQ(getDataFlowGraphNumChildren(varAssignmSum), 2);
  // [VarAssignm] i= i + 1;   (For update)
  auto varAssignmI = varAssignmSum->getControlFlowGraph()->getOnlyChild();
  EXPECT_EQ(getDataFlowGraphNumParents(varAssignmI), 2);
  EXPECT_EQ(getDataFlowGraphNumChildren(varAssignmI), 3);

  // check existence of expected data flow graph edges
  // inputA
  EXPECT_TRUE(checkDataFlowGraphEdge(parameterList, logicalExpr));
  // base
  EXPECT_TRUE(checkDataFlowGraphEdge(varDeclBase, varAssignmSum));
  // sum
  EXPECT_TRUE(checkDataFlowGraphEdge(varDeclSum, returnStmt));
  EXPECT_TRUE(checkDataFlowGraphEdge(varDeclSum, varAssignmSum));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignmSum, varAssignmSum));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignmSum, returnStmt));
  // i
  EXPECT_TRUE(checkDataFlowGraphEdge(varDeclI, logicalExpr));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignmI, logicalExpr));
  EXPECT_TRUE(checkDataFlowGraphEdge(varDeclI, varAssignmSum));
  EXPECT_TRUE(checkDataFlowGraphEdge(varAssignmI, varAssignmSum));
  EXPECT_TRUE(checkDataFlowGraphEdge(varDeclI, varAssignmI));
}
