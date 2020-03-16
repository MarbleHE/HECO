#include <ControlFlowGraphVisitor.h>
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"

class FlowGraphVisitorFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // This is required because some methods in ControlFlowGraphVisitor rely on the unique node ID which differs
    // depending on the order in which the tests are executed, therefore we need to reset the ongoing ID counter before
    // each test run.
    AbstractNode::resetNodeIdCounter();
  }
};

// Expected control flow graph:
//
//              Function_0
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

  auto function0 = new GraphNode();
  auto block2 = new GraphNode({function0});
  auto if35 = new GraphNode({block2});
  auto block34 = new GraphNode({if35});
  auto varAssignm33 = new GraphNode({block34});
  auto block27 = new GraphNode({if35});
  auto varAssignm26 = new GraphNode({block27});
  auto if45 = new GraphNode({varAssignm26, varAssignm33});
  auto varAssignm44 = new GraphNode({if45});
  auto return50 = new GraphNode({varAssignm44});

  EXPECT_TRUE(GraphNode::areEqualGraphs(function0, fgv.getRootNode()));
}

TEST_F(FlowGraphVisitorFixture, traverseAndPrintNodeTest) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(9, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  std::stringstream generatedOutput;
  GraphNode::traverseAndPrintNodes(*(fgv.getRootNode()), generatedOutput);

  std::string expectedOutput = "(1) Function_0\n"
                               "(1) \tBlock_2\n"
                               "(2) \t\tIf_35\n"
                               "(1) \t\t\tBlock_27\n"
                               "(1) \t\t\t\tVarAssignm_26\n"
                               "(1) \t\t\t\t\tIf_45\n"
                               "(1) \t\t\t\t\t\tVarAssignm_44\n"
                               "(0) \t\t\t\t\t\t\tReturn_50\n"
                               "(1) \t\t\tBlock_34\n"
                               "(1) \t\t\t\tVarAssignm_33\n"
                               "(1) \t\t\t\t\tIf_45\n"
                               "    \t\t\t\t\t... see above, visiting an already visited node ...\n";

  EXPECT_EQ(expectedOutput, generatedOutput.str());
}

TEST_F(FlowGraphVisitorFixture, controlFlowGraphIncludingWhileStatement) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(11, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  auto function0 = new GraphNode();
  auto block2 = new GraphNode({function0});
  auto varDecl17 = new GraphNode({block2});
  auto varDecl19 = new GraphNode({varDecl17});
  auto while49 = new GraphNode({varDecl19});
  auto block48 = new GraphNode({while49});
  auto varAssignm41 = new GraphNode({block48});
  auto varAssignm47 = new GraphNode({varAssignm41});
  while49->addParent(varAssignm47);
  auto return51 = new GraphNode({while49});

  EXPECT_TRUE(GraphNode::areEqualGraphs(function0, fgv.getRootNode()));
}

// Commented output of traverseAndPrintNodes:
//
// (1)  Function_0 (sumNTimes2)
// (1)     Block_2 ({ ... })
// (1)        VarDecl_9 (int sum = 0;)
// (1)           VarDecl_13 (int base = 2;)
// (1)              For_38 (for ...)
// (2)                 VarDecl_17 (int i = 0; [For initializer])
// (1)                    Block_37 ({ ... }, [For body])
// (1)                       VarAssignm_36 (sum = sum + base * i; [For body statement 1])
// (2)                          VarAssignm_28 (i=i+1; [For update statement])
// (1)                             Block_37 ({ ... }, [For body])
//                                 ... see above, visiting an already visited node ...
// (0)                                Return_40
// (0)                       Return_40
TEST_F(FlowGraphVisitorFixture, controlFlowGraphIncludingForStatement) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(23, ast);

  ControlFlowGraphVisitor fgv;
  fgv.visit(ast);

  auto function0 = new GraphNode();
  auto block2 = new GraphNode({function0});
  auto varDecl9 = new GraphNode({block2});
  auto varDecl13 = new GraphNode({varDecl9});
  auto for38 = new GraphNode({varDecl13});
  auto varDecl17 = new GraphNode({for38});
  auto block37 = new GraphNode({varDecl17});
  auto varAssignm36 = new GraphNode({block37});
  auto varAssignm28 = new GraphNode({varAssignm36});
  block37->addParent(varAssignm28);
  auto return40 = new GraphNode({varAssignm28, varDecl17});

  EXPECT_TRUE(GraphNode::areEqualGraphs(function0, fgv.getRootNode()));
}
