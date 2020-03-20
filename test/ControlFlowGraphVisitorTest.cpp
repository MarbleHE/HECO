#include "ControlFlowGraphVisitor.h"
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"
#include "GraphNode.h"

class FlowGraphVisitorFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // This is required because some methods in ControlFlowGraphVisitor rely on the unique node ID which differs
    // depending on the order in which the tests are executed, therefore we need to reset the ongoing ID counter before
    // each test run.
    AbstractNode::resetNodeIdCounter();
  }

  static bool checkReadWrites(std::initializer_list<std::string> readVariables,
                              std::initializer_list<std::string> writeVariables, GraphNode *gNode) {
    // generate sets out of given read/write variables to have fast O(1) access
    std::set<std::string> reads(readVariables.begin(), readVariables.end());
    std::set<std::string> writes(writeVariables.begin(), writeVariables.end());
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
  auto return50 = new GraphNode(relType, {varAssignm44});

  EXPECT_TRUE(function0->getControlFlowGraph()->areEqualGraphs(fgv.getRootNodeCfg()));
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
                               "(1) \t\t\t\t\t\t\t\tLogicalExpr_39\n"
                               "(1) \t\t\t\t\t\t\t\t\tVarAssignm_44\n"
                               "(0) \t\t\t\t\t\t\t\t\t\tReturn_50\n"
                               "(1) \t\t\t\t\tBlock_34\n"
                               "(1) \t\t\t\t\t\tVarAssignm_33\n"
                               "(1) \t\t\t\t\t\t\tIf_45\n"
                               "    \t\t\t\t\t\t\t... see above, visiting an already visited node ...\n";

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

  EXPECT_TRUE(function0->getControlFlowGraph()->areEqualGraphs(fgv.getRootNodeCfg()));
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
  auto return40 = new GraphNode(relType, {logicalExpr21});

  EXPECT_TRUE(function0->getControlFlowGraph()->areEqualGraphs(fgv.getRootNodeCfg()));
}

TEST_F(FlowGraphVisitorFixture, dataflowGraph_detectedVariableReadWrites) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(23, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  // [Function] sumNTimes2
  auto cfgRootNode = fgv.getRootNodeCfg();
  EXPECT_TRUE(checkReadWrites({}, {}, cfgRootNode));

  // [ParameterList] (int inputA)
  auto parameterList = cfgRootNode->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkReadWrites({}, {"inputA"}, parameterList));

  // [VarDecl] int sum = 0;
  auto varDeclSum = parameterList->getControlFlowGraph()->getOnlyChild()->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkReadWrites({}, {"sum"}, varDeclSum));

  // [VarDecl] int base = 2;
  auto varDeclBase = varDeclSum->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkReadWrites({}, {"base"}, varDeclBase));

  // [For] for (initializer; [LogicalExpr] i <= inputA; updateStatement)
  auto forStmt = varDeclBase->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkReadWrites({}, {}, forStmt));

  // [VarDecl] int i = 0;
  auto varDeclI = forStmt->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkReadWrites({}, {"i"}, varDeclI));

  // [LogicalExpr] i <= inputA
  auto logicalExpr = varDeclI->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkReadWrites({"i", "inputA"}, {}, logicalExpr));

  // [Return] return sum;
  auto returnStmt = logicalExpr->getControlFlowGraph()->getChildAtIndex(1);
  EXPECT_TRUE(checkReadWrites({"sum"}, {}, returnStmt));

  // [VarAssignm] sum = logicalExpr + base * i;
  auto varAssignmSum = logicalExpr->getControlFlowGraph()->getChildAtIndex(0)->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkReadWrites({"sum", "base", "i"}, {"sum"}, varAssignmSum));

  // [VarAssignm] i= i + 1;
  auto varAssignmI = varAssignmSum->getControlFlowGraph()->getOnlyChild();
  EXPECT_TRUE(checkReadWrites({"i"}, {"i"}, varAssignmI));
}
