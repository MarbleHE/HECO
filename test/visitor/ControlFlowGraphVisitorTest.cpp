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

  auto forBlock = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({forInitializer.get()}));
  auto varAssignmStmt2 = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({forBlock.get()}));
  auto forUpdate = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({varAssignmStmt2.get()}));
  varAssignmStmt2->getControlFlowGraph().addChild(*forUpdate);

  auto returnStmt = std::make_unique<GraphNode>(dummyAstNode, relType, createParentsVector({forUpdate.get()}));

  // check that the CFG's structure is correct
  EXPECT_TRUE(functionStmt->getControlFlowGraph().isEqualToGraph(cfgv.getRootNode()));

  // check that all nodes of the CFG have a refToOriginalNode set
  auto allNodes = cfgv.getRootNode().getControlFlowGraph().getAllReachableNodes();
  // Note that we added 1 because the Parser always wraps the parsed statements into a Block.
  // Also, note that initializer and update of the For loop are wrapped into blocks each.
  EXPECT_EQ(allNodes.size(), 13);
}


// == Tests for the Data Flow Graph
// We consider two types of tests here: (1) Tests that check whether the accessedVariables map contains the correct
// information about read/written variables. (2) Tests that verify that the graph structure of the data flow graph
// matches the expected one (similar as for the control flow graph).
// Tests for category (1) are built during the first pass (in which the CFG is built too) and do not require to build
// the data flow graph at all.

// (1) Tests for the accessedVariables map

bool setContains(const std::set<VariableAccessPair> &variableAccesses,
                 const std::string &expectedIdentifier,
                 VariableAccessType expectedAccessType) {
  for (auto[scopeIdentifier, accessType] : variableAccesses) {
    if (scopeIdentifier.getId()==expectedIdentifier
        && accessType==expectedAccessType) {
      return true;
    }
  }
  return false;
}

GraphNode &getNthStatementInBlock(ControlFlowGraphVisitor &cfgv, int n) {
  return cfgv.getRootNode().getDataFlowGraph().getChildAtIndex(n);
}

TEST(ControlFlowGraphVisitorTest, dfg_noScopeGiven_expectFail) { /* NOLINT */
  const char *inputChars = R""""(
    int z = 0;
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  EXPECT_THROW(inputAST->accept(cfgv), std::runtime_error);
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
  inputAST->accept(cfgv);

  // get first child as Block (is the root node)
  auto accessedVariables = getNthStatementInBlock(cfgv, 0).getAccessedVariables();

  EXPECT_EQ(accessedVariables.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables, "z", VariableAccessType::WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_simpleReadWriteAssignment) { /* NOLINT */
  const char *inputChars = R""""(
    {
      int q = a + 7;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVariables = getNthStatementInBlock(cfgv, 0).getAccessedVariables();

  EXPECT_EQ(accessedVariables.size(), 2);
  EXPECT_TRUE(setContains(accessedVariables, "q", VariableAccessType::WRITE));
  EXPECT_TRUE(setContains(accessedVariables, "a", VariableAccessType::READ));
}

TEST(ControlFlowGraphVisitorTest, dfg_ifStatement) { /* NOLINT */
  const char *inputChars = R""""(
    {
      if (c > 100) {
        a = 22;
      }
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVariables_ifStatement = getNthStatementInBlock(cfgv, 0).getAccessedVariables();
  EXPECT_EQ(accessedVariables_ifStatement.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_ifStatement, "c", VariableAccessType::READ));

  auto accessedVariables_thenBlock =
      getNthStatementInBlock(cfgv, 0).getDataFlowGraph().getChildAtIndex(1).getAccessedVariables();
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

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVariables_ifStatement = getNthStatementInBlock(cfgv, 0).getAccessedVariables();
  EXPECT_EQ(accessedVariables_ifStatement.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_ifStatement, "c", VariableAccessType::READ));

  auto accessedVariables_thenBlock =
      getNthStatementInBlock(cfgv, 0).getDataFlowGraph().getChildAtIndex(1).getAccessedVariables();
  EXPECT_EQ(accessedVariables_thenBlock.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_thenBlock, "a", VariableAccessType::WRITE));

  auto accessedVariables_elseBlock =
      getNthStatementInBlock(cfgv, 0).getDataFlowGraph().getChildAtIndex(2).getAccessedVariables();
  EXPECT_EQ(accessedVariables_thenBlock.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_thenBlock, "a", VariableAccessType::WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_forLoop_accumulation) { /* NOLINT */
  const char *inputChars = R""""(
    {
      int sum = 0;
      for (int i = 0; i < 100; ++i) {
        sum = sum + 1;
      }
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVariables_initializer =
      getNthStatementInBlock(cfgv, 1).getDataFlowGraph().getChildAtIndex(0).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "i", VariableAccessType::WRITE));

  auto accessedVariables_condition =
      getNthStatementInBlock(cfgv, 1).getDataFlowGraph().getChildAtIndex(1).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "i", VariableAccessType::READ));

  auto accessedVariables_update =
      getNthStatementInBlock(cfgv, 1).getDataFlowGraph().getChildAtIndex(2).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "i", VariableAccessType::WRITE));

  auto accessedVariables_block =
      getNthStatementInBlock(cfgv, 1).getDataFlowGraph().getChildAtIndex(3).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 2);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "sum", VariableAccessType::READ));
  EXPECT_TRUE(setContains(accessedVariables_initializer, "sum", VariableAccessType::WRITE));
}

TEST(ControlFlowGraphVisitorTest, dfg_forLoop_localVariable_emptyUpdate) { /* NOLINT */
  // TODO: Remove outer block
  // TODO: ControlFlowGraphVisitor akzeptiert als Eingabe Block/If/For
  const char *inputChars = R""""(
    {
      for (int i = 0; i < 100; ) {
        int c = i+1;
      }
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ControlFlowGraphVisitor cfgv;
  inputAST->accept(cfgv);

  auto accessedVariables_initializer =
      getNthStatementInBlock(cfgv, 1).getDataFlowGraph().getChildAtIndex(0).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "i", VariableAccessType::WRITE));

  auto accessedVariables_condition =
      getNthStatementInBlock(cfgv, 1).getDataFlowGraph().getChildAtIndex(1).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 1);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "i", VariableAccessType::READ));

  auto accessedVariables_update =
      getNthStatementInBlock(cfgv, 1).getDataFlowGraph().getChildAtIndex(2).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 0);

  auto accessedVariables_block =
      getNthStatementInBlock(cfgv, 1).getDataFlowGraph().getChildAtIndex(3).getAccessedVariables();
  EXPECT_EQ(accessedVariables_initializer.size(), 2);
  EXPECT_TRUE(setContains(accessedVariables_initializer, "i", VariableAccessType::READ));
  EXPECT_TRUE(setContains(accessedVariables_initializer, "c", VariableAccessType::WRITE));
}

// TODO: add many more test programs


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
//}
