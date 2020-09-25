#include <include/ast_opt/utilities/Datatype.h>
#include <include/ast_opt/parser/Parser.h>
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "include/ast_opt/visitor/ControlFlowGraph/ControlFlowGraphVisitor.h"
#include "gtest/gtest.h"

// == Tests for the Control Flow Graph

std::vector<std::reference_wrapper<GraphNode>> createParentsVector(std::initializer_list<GraphNode *> nodes) {
  std::vector<std::reference_wrapper<GraphNode>> parents;
  for (auto node : nodes) parents.push_back(std::ref(*node));
  return parents;
}

TEST(ControlFlowGraphVisitorTest, cfg_simpleProgram) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(int a, int z, int v) {
      int a = 10;
      return a;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto &gn = cfgv.getRootNode();

  const auto relType = RelationshipType::CTRL_FLOW_GRAPH;
  auto dummyAstNode = Block();
  auto functionStmt = std::make_unique<GraphNode>(dummyAstNode);
  auto blockStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({functionStmt.get()}));
  auto varDeclStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({blockStmt.get()}));
  auto returnStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varDeclStmt.get()}));

  // check that the CFG's structure is correct
  EXPECT_TRUE(functionStmt->getControlFlowGraph().isEqualToGraph(cfgv.getRootNode()));

  // check that all nodes of the CFG have a refToOriginalNode set
  auto allNodes = cfgv.getRootNode().getControlFlowGraph().getAllReachableNodes();
  // Note that we added 1 because the Parser always wraps the parsed statements into a Block
  EXPECT_EQ(allNodes.size(), 5);
}

TEST(ControlFlowGraphVisitorTest, cfg_ifElseProgram) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(int a) {
      int q;
      q = 21;
      if (q > a) {
        return 1;
      } else {
        return 0;
      }
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto &gn = cfgv.getRootNode();

  const auto relType = RelationshipType::CTRL_FLOW_GRAPH;
  auto dummyAstNode = Block();
  auto functionStmt = std::make_unique<GraphNode>(dummyAstNode);
  auto blockStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({functionStmt.get()}));
  auto varDeclStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({blockStmt.get()}));
  auto varAssignmStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varDeclStmt.get()}));
  auto ifAssignmStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varAssignmStmt.get()}));
  auto thenBranch = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({ifAssignmStmt.get()}));
  auto thenReturnStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({thenBranch.get()}));
  auto elseBranch = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({ifAssignmStmt.get()}));
  auto elseReturnStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({elseBranch.get()}));

  // check that the CFG's structure is correct
  EXPECT_TRUE(functionStmt->getControlFlowGraph().isEqualToGraph(cfgv.getRootNode()));

  // check that all nodes of the CFG have a refToOriginalNode set
  auto allNodes = cfgv.getRootNode().getControlFlowGraph().getAllReachableNodes();
  // Note that we added 1 because the Parser always wraps the parsed statements into a Block
  EXPECT_EQ(allNodes.size(), 10);
}

TEST(ControlFlowGraphVisitorTest, cfg_ifProgram) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(int a) {
      int q;
      q = 21;
      if (q > a) {
        return 1;
      }
      return 0;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto &gn = cfgv.getRootNode();

  const auto relType = RelationshipType::CTRL_FLOW_GRAPH;
  auto dummyAstNode = Block();
  auto functionStmt = std::make_unique<GraphNode>(dummyAstNode);
  auto blockStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({functionStmt.get()}));
  auto varDeclStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({blockStmt.get()}));
  auto varAssignmStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varDeclStmt.get()}));
  auto ifAssignmStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varAssignmStmt.get()}));
  auto thenBranch = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({ifAssignmStmt.get()}));
  auto thenReturnStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({thenBranch.get()}));
  auto returnStmt2 = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({ifAssignmStmt.get()}));

  // check that the CFG's structure is correct
  EXPECT_TRUE(functionStmt->getControlFlowGraph().isEqualToGraph(cfgv.getRootNode()));

  // check that all nodes of the CFG have a refToOriginalNode set
  auto allNodes = cfgv.getRootNode().getControlFlowGraph().getAllReachableNodes();
  // Note that we added 1 because the Parser always wraps the parsed statements into a Block
  EXPECT_EQ(allNodes.size(), 9);
}

TEST(ControlFlowGraphVisitorTest, cfg_ForProgram) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(int a) {
      int q;
      q = 21;
      for (int i = 0; i < 22; i = i+1) {
        q = q + i * 12;
      }
      return q;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto &gn = cfgv.getRootNode();

  const auto relType = RelationshipType::CTRL_FLOW_GRAPH;
  auto dummyAstNode = Block();
  auto functionStmt = std::make_unique<GraphNode>(dummyAstNode);
  auto blockStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({functionStmt.get()}));
  auto varDeclStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({blockStmt.get()}));
  auto varAssignmStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varDeclStmt.get()}));
  auto forStatement = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varAssignmStmt.get()}));
  auto forInitializer = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({forStatement.get()}));
  auto forCondition = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({forInitializer.get()}));

  auto forBlock = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({forCondition.get()}));
  auto varAssignmStmt2 = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({forBlock.get()}));
  auto forUpdate = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varAssignmStmt2.get()}));
  forCondition->getControlFlowGraph().addParent(*forUpdate);

  auto returnStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({forCondition.get()}));

  // check that the CFG's structure is correct
  EXPECT_TRUE(functionStmt->getControlFlowGraph().isEqualToGraph(cfgv.getRootNode()));

  // check that all nodes of the CFG have a refToOriginalNode set
  auto allNodes = cfgv.getRootNode().getControlFlowGraph().getAllReachableNodes();
  // Note that we added 1 because the Parser always wraps the parsed statements into a Block.
  // Also, note that initializer and update of the For loop are wrapped into blocks each but we do not visit these
  // blocks and as such they are not counted.
  EXPECT_EQ(allNodes.size(), 12);
}


// == Tests for the Data Flow Graph
// We consider two types of tests here: (1) Tests that check whether the accessedVariables map contains the correct
// information about read/written variables. (2) Tests that verify that the graph structure of the data flow graph
// matches the expected one (similar as for the control flow graph).
// Tests for category (1) are built during the first pass (in which the CFG is built too) and do not require to build
// the data flow graph at all.

// (1) Tests for the accessedVariables map

bool setContains(const VarAccessMapType &variableAccesses,
                 const std::string &expectedIdentifier,
                 VariableAccessType expectedAccessType) {
  auto it = std::find_if(variableAccesses.begin(),
                         variableAccesses.end(),
                         [&expectedIdentifier, &expectedAccessType](const auto &p) {
                           return p.first.getId()==expectedIdentifier
                               && p.second==expectedAccessType;
                         });
  return (it!=variableAccesses.end());
}

GraphNode &getGraphNodeByChildrenIdxPath(GraphNode &graphNode,
                                         std::initializer_list<int> childrenPath,
                                         RelationshipType relationshipType = RelationshipType::CTRL_FLOW_GRAPH) {
  // this test helper simplifies accessing nested graph nodes
  // for example childrenPath = {0, 1, 4, 1} means
  //    graphNode.getChildren(0)->getChildren(1)->getChildren(4)->getChildren(1)
  // where children refer to the children in the graph corresponding to the given RelationshipType.
  GraphNode *curNode = &graphNode;
  for (auto &idx : childrenPath) {
    if (relationshipType==RelationshipType::CTRL_FLOW_GRAPH) {
      curNode = &curNode->getControlFlowGraph().getChildAtIndex(idx);
    } else if (relationshipType==RelationshipType::DATA_FLOW_GRAPH) {
      curNode = &curNode->getDataFlowGraph().getChildAtIndex(idx);
    }
  }
  return *curNode;
}

TEST(ControlFlowGraphVisitorTest, dfg_noScopeGiven_expectFail) { /* NOLINT */
  const char *inputChars = R""""(
    int z = 0;
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  // we need to use inputAst->begin() here as the parser wraps everything into a Block
  EXPECT_THROW(inputAST->begin()->accept(cfgv), std::runtime_error);
}

TEST(ControlFlowGraphVisitorTest, dfg_simpleAssignment) { /* NOLINT */
  const char *inputChars = R""""(
    {
      int z = 0;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->begin()->accept(cfgv);

  // get first child in Block as the root node is a Block
  auto accessedVariables = getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0}).getAccessedVariables();

  EXPECT_EQ(accessedVariables.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables, "z", VariableAccessType::WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_functionParameter) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(int z) {
      int v = z+1;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVars_function = getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0}).getAccessedVariables();
  EXPECT_EQ(accessedVars_function.size(), 1);
  EXPECT_TRUE(setContains(accessedVars_function, "z", VariableAccessType::WRITE));

  // get first child in Block as the root node is a Block
  auto accessedVariables = getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables.size(), 2);
  EXPECT_TRUE(setContains(accessedVariables, "z", VariableAccessType::READ));
  EXPECT_TRUE(setContains(accessedVariables, "v", VariableAccessType::WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_simpleReadWriteAssignment_throwErrorOnNonResolvableVariable) { /* NOLINT */
  const char *inputChars = R""""(
    {
      int q = a + 7;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  EXPECT_THROW(inputAST->begin()->accept(cfgv), std::runtime_error);
}

TEST(ControlFlowGraphVisitorTest, dfg_simpleReadWriteAssignment_ignoreNonResolvedVars) { /* NOLINT */
  const char *inputChars = R""""(
    {
      int q = a + 7;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv(true);  // this flag let's the visitor ignore non-resolvable variables
  inputAST->begin()->accept(cfgv);

  auto accessedVariables = getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0}).getAccessedVariables();

  // Variable "a" is declared out-of-scope and due to the flag passed to the cfgv constructor its accesses are not
  // tracked by the visitor.
  EXPECT_EQ(accessedVariables.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables, "q", VariableAccessType::WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_simpleReadWriteAssignment_takeOutOfScopeDeclaredVarsAsInput) { /* NOLINT */
  const char *inputChars = R""""(
    {
      int q = a + 7;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  std::unordered_set<std::unique_ptr<ScopedIdentifier>, ScopedIdentifierHashFunction> identifiers;

  // here we define which variables were declared in some parent node of the given AST
  std::vector<std::string> predeclaredVariables = {"a"};
  ControlFlowGraphVisitor cfgv(predeclaredVariables);
  inputAST->begin()->accept(cfgv);

  auto accessedVariables = getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0}).getAccessedVariables();

  EXPECT_EQ(accessedVariables.size(), 2);
  EXPECT_TRUE(setContains(accessedVariables, "q", VariableAccessType::WRITE));
  EXPECT_TRUE(setContains(accessedVariables, "a", VariableAccessType::READ));
}

TEST(ControlFlowGraphVisitorTest, dfg_ifStatement) { /* NOLINT */
  const char *inputChars = R""""(
  {
      int c = 99;
      int a;
      if (c > 100) {
        a = 22;
      }
  }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVariables_ifStatement =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_ifStatement.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_ifStatement, "c", VariableAccessType::READ));

  auto accessedVariables_thenBlock =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0, 0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_thenBlock.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_thenBlock, "a", VariableAccessType::WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_ifElseStatement) { /* NOLINT */
  const char *inputChars = R""""(
    {
      if (c > 100) {
        a = 22;
      } else {
        a = 43;
      }
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  std::vector<std::string> predeclaredVars = {"c", "a"};
  ControlFlowGraphVisitor cfgv(predeclaredVars);
  inputAST->begin()->accept(cfgv);

  auto accessedVariables_ifStatement = getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_ifStatement.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_ifStatement, "c", VariableAccessType::READ));

  auto accessedVariables_thenBlock =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_thenBlock.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_thenBlock, "a", VariableAccessType::WRITE));

  auto accessedVariables_elseBlock =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 1, 0}).getAccessedVariables();;
  EXPECT_EQ(accessedVariables_thenBlock.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_thenBlock, "a", VariableAccessType::WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_forLoop_accumulation) { /* NOLINT */
  const char *inputChars = R""""(
    {
      int sum = 0;
      for (int i = 0; i < 100; i=i+1) {
        sum = sum + 1;
      }
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVariables_initializer =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "i", VariableAccessType::WRITE));

  auto accessedVariables_block =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0, 0, 0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_block.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_block, "sum", VariableAccessType::READ_AND_WRITE));

  auto accessedVariables_update =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0, 0, 0, 0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_update.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_update, "i", VariableAccessType::READ_AND_WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_forLoop_localVariable_emptyUpdate) { /* NOLINT */
  const char *inputChars = R""""(
      for (int i = 0; i < 100; ) {
        int c = i+1;
      }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVariables_initializer =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0}).getAccessedVariables();

  EXPECT_EQ(accessedVariables_initializer.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "i", VariableAccessType::WRITE));

  auto accessedVariables_condition =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_condition.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_condition, "i", VariableAccessType::READ));

  auto accessedVariables_block =
      getGraphNodeByChildrenIdxPath(cfgv.getRootNode(), {0, 0, 0, 0, 0}).getAccessedVariables();
  EXPECT_EQ(accessedVariables_block.size(), 2);
  EXPECT_TRUE(setContains(accessedVariables_block, "i", VariableAccessType::READ));
  EXPECT_TRUE(setContains(accessedVariables_block, "c", VariableAccessType::WRITE));
}

// (2) Tests for the structure accessedVariables map

TEST(ControlFlowGraphVisitorTest, dfgGraph_simpleAssignment) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(int a) {
      a = a + 34;
      return a;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto &gn = cfgv.getRootNode();

  cfgv.buildDataflowGraph();

//  std::cout << "\n== DATA FLOW GRAPH =======" << std::endl;
//
//  for (auto &dfn : cfgv.getRootNode().getControlFlowGraph().getAllReachableNodes()) {
//    if (!dfn.get().getControlFlowGraph().getChildren().empty()
//        || !dfn.get().getControlFlowGraph().getParents().empty()) {
//      std::cout << "Node: " << dfn.get().getAstNode().getUniqueNodeId() << std::endl;
//      std::cout << "\tchildren: " << std::endl;
//      for (auto &child : dfn.get().getDataFlowGraph().getChildren()) {
//        std::cout << "\t– " << child.get().getAstNode().getUniqueNodeId() << std::endl;
//      }
//      std::cout << std::endl;
//      std::cout << "\tparents: " << std::endl;
//      for (auto &parent : dfn.get().getDataFlowGraph().getParents()) {
//        std::cout << "\t– " << parent.get().getAstNode().getUniqueNodeId() << std::endl;
//      }
//      std::cout << std::endl;
//    }
//  }

  auto &functionStmt = getGraphNodeByChildrenIdxPath(gn, {0});
  auto &varAssignm = getGraphNodeByChildrenIdxPath(gn, {0, 0, 0});
  auto &returnStmt = getGraphNodeByChildrenIdxPath(gn, {0, 0, 0, 0});

  EXPECT_EQ(varAssignm.getDataFlowGraph().getParents().size(), 1);
  EXPECT_EQ(&varAssignm.getDataFlowGraph().getParents().at(0).get(), &functionStmt);
  EXPECT_EQ(varAssignm.getDataFlowGraph().getChildren().size(), 1);
  EXPECT_EQ(&varAssignm.getDataFlowGraph().getChildren().at(0).get(), &returnStmt);
//
  EXPECT_EQ(returnStmt.getDataFlowGraph().getParents().size(), 1);
  EXPECT_EQ(&returnStmt.getDataFlowGraph().getParents().at(0).get(), &varAssignm);
  EXPECT_EQ(returnStmt.getDataFlowGraph().getChildren().size(), 0);

}


//TEST(ControlFlowGraphVisitorTest, wip) {
//  // Confirm that printing children works as expected
//
//  // int scalar = {2};
//  auto declarationScalar = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
//                                                                 std::move(std::make_unique<Variable>("scalar")),
//                                                                 std::move(std::make_unique<LiteralInt>(2)));
//
//  // int vec = {3, 4, 9, 2, 1};
//  std::vector<std::unique_ptr<AbstractExpression>> exprs;
//  exprs.emplace_back(std::make_unique<LiteralInt>(3));
//  exprs.emplace_back(std::make_unique<LiteralInt>(4));
//  exprs.emplace_back(std::make_unique<LiteralInt>(9));
//  exprs.emplace_back(std::make_unique<LiteralInt>(2));
//  exprs.emplace_back(std::make_unique<LiteralInt>(1));
//  auto declarationVec = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
//                                                              std::make_unique<Variable>("vec"),
//                                                              std::make_unique<ExpressionList>(std::move(exprs)));
//
//  // public void main() { ... }
//  std::vector<std::unique_ptr<AbstractStatement>> statements;
//  statements.push_back(std::move(declarationScalar));
//  statements.push_back(std::move(declarationVec));
//  auto statementBlock = std::make_unique<Block>(std::move(statements));
//  auto expected = std::make_unique<Function>(Datatype(Type::VOID),
//                                             "main",
//                                             std::move(std::vector<std::unique_ptr<FunctionParameter>>()),
//                                             std::move(statementBlock));
//
////  std::stringstream ss;
//  ControlFlowGraphVisitor cfgv;
//  cfgv.visit(*expected);
//
//  std::cout << "test" << std::endl;
//
////  EXPECT_EQ(ss.str(), "Assignment\n"
////                      "  Variable (foo)\n"
////                      "  LiteralBool (true)\n");

