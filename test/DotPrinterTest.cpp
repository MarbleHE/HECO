#include <Node.h>
#include <Operator.h>
#include <Variable.h>
#include <BinaryExpr.h>
#include <Function.h>
#include <include/utilities/DotPrinter.h>
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"

class DotPrinterFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // This is required, otherwise the node IDs won't match the expected result as they are incremented ongoing but
    // must be reset after each test case.
    Node::resetNodeIdCounter();
  }

  static Node* buildBinaryExpression() {
    return new BinaryExpr(
        new Variable("alpha"),
        OpSymb::multiplication,
        new LiteralInt(212));
  }
};

TEST_F(DotPrinterFixture, getDotFormattedStringTest_printSimpleBinaryExpression) {  /* NOLINT */
  auto binaryExpression = buildBinaryExpression();

  auto expectedStr =
      "  BinaryExpr_0 [label=\"BinaryExpr_0\\n[l(v): 0, r(v): 0]\" shape=oval style=filled fillcolor=white]\n"
      "  { BinaryExpr_0 } -> { Variable_1, Operator_2, LiteralInt_3 }\n";

  DotPrinter dp;
  dp.setIndentationCharacter("  ").setShowMultDepth(true);

  ASSERT_EQ(dp.getDotFormattedString(binaryExpression), expectedStr);
}

TEST_F(DotPrinterFixture, getDotFormattedStringTest_printReversedBinaryExpression) {  /* NOLINT */
  auto binaryExpression = buildBinaryExpression();

  // reversing the edge should only flip parents with children
  binaryExpression->swapChildrenParents();

  auto expectedStr =
      "\tBinaryExpr_0 [label=\"BinaryExpr_0\\n[l(v): 0, r(v): 0]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ Variable_1, Operator_2, LiteralInt_3 } -> { BinaryExpr_0 }\n";

  DotPrinter dp;
  dp.setIndentationCharacter("\t").setShowMultDepth(true);

  ASSERT_EQ(dp.getDotFormattedString(binaryExpression), expectedStr);
}

TEST_F(DotPrinterFixture, getDotFormattedStringTest_FunctionNotCircuitCompatibleException) {  /* NOLINT */
  auto function = new Function("abc");

  DotPrinter dp;
  ASSERT_THROW(dp.printAllReachableNodes(function), std::logic_error);
}

TEST_F(DotPrinterFixture, printAsDotFormattedGraphTest_printAstExample1) {  /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(18, ast);

  std::stringstream outputStream;
  DotPrinter dp;
  dp.setShowMultDepth(true).setOutputStream(outputStream);
  dp.printAsDotFormattedGraph(ast);

  auto expectedStr =
      "digraph D {\n"
      "\tReturn_0 [label=\"Return_0\\n[l(v): 2, r(v): 0]\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Return_0 } -> { LogicalExpr_1 }\n"
      "\tLogicalExpr_1 [label=\"LogicalExpr_1\\n[l(v): 2, r(v): 0]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_1 } -> { LogicalExpr_2, Operator_3, Variable_4 }\n"
      "\tVariable_4 [label=\"Variable_4\\n[l(v): 0, r(v): 1]\\na_t\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_3 [label=\"Operator_3\\n[l(v): 0, r(v): 1]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_2 [label=\"LogicalExpr_2\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_2 } -> { LogicalExpr_5, Operator_6, Variable_7 }\n"
      "\tVariable_7 [label=\"Variable_7\\n[l(v): 0, r(v): 1]\\ny_1\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_6 [label=\"Operator_6\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_5 [label=\"LogicalExpr_5\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_5 } -> { LogicalExpr_8, Operator_9, LogicalExpr_10 }\n"
      "\tLogicalExpr_10 [label=\"LogicalExpr_10\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_10 } -> { Variable_11, Operator_12, Variable_13 }\n"
      "\tVariable_13 [label=\"Variable_13\\n[l(v): 0, r(v): 2]\\na_2^(2)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_12 [label=\"Operator_12\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_11 [label=\"Variable_11\\n[l(v): 0, r(v): 2]\\na_1^(2)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_9 [label=\"Operator_9\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_8 [label=\"LogicalExpr_8\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_8 } -> { Variable_14, Operator_15, Variable_16 }\n"
      "\tVariable_16 [label=\"Variable_16\\n[l(v): 0, r(v): 2]\\na_2^(1)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_15 [label=\"Operator_15\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_14 [label=\"Variable_14\\n[l(v): 0, r(v): 2]\\na_1^(1)\" shape=oval style=filled fillcolor=white]\n"
      "}\n";

  ASSERT_EQ(outputStream.str(), expectedStr);
}

TEST_F(DotPrinterFixture, printAsDotFormattedGraphTest_printAstExample2) {  /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(19, ast);

  std::stringstream outputStream;
  DotPrinter dp;
  dp.setShowMultDepth(true)
      .setOutputStream(outputStream);
  dp.printAsDotFormattedGraph(ast);

  auto expectedStr =
      "digraph D {\n"
      "\tReturn_0 [label=\"Return_0\\n[l(v): 3, r(v): 0]\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Return_0 } -> { LogicalExpr_1 }\n"
      "\tLogicalExpr_1 [label=\"LogicalExpr_1\\n[l(v): 3, r(v): 0]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_1 } -> { LogicalExpr_2, Operator_3, Variable_4 }\n"
      "\tVariable_4 [label=\"Variable_4\\n[l(v): 0, r(v): 1]\\na_t\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_3 [label=\"Operator_3\\n[l(v): 0, r(v): 1]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_2 [label=\"LogicalExpr_2\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_2 } -> { LogicalExpr_5, Operator_6, Variable_7 }\n"
      "\tVariable_7 [label=\"Variable_7\\n[l(v): 0, r(v): 1]\\ny_1\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_6 [label=\"Operator_6\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_5 [label=\"LogicalExpr_5\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_5 } -> { LogicalExpr_8, Operator_9, LogicalExpr_10 }\n"
      "\tLogicalExpr_10 [label=\"LogicalExpr_10\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_10 } -> { LogicalExpr_11, Operator_12, LogicalExpr_13 }\n"
      "\tLogicalExpr_13 [label=\"LogicalExpr_13\\n[l(v): 0, r(v): 2]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_13 } -> { Variable_14, Operator_15, Variable_16 }\n"
      "\tVariable_16 [label=\"Variable_16\\n[l(v): 0, r(v): 2]\\na_2^(2)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_15 [label=\"Operator_15\\n[l(v): 0, r(v): 2]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_14 [label=\"Variable_14\\n[l(v): 0, r(v): 2]\\na_2^(2)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_12 [label=\"Operator_12\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_11 [label=\"LogicalExpr_11\\n[l(v): 1, r(v): 2]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_11 } -> { Variable_17, Operator_18, Variable_19 }\n"
      "\tVariable_19 [label=\"Variable_19\\n[l(v): 0, r(v): 3]\\na_1^(2)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_18 [label=\"Operator_18\\n[l(v): 0, r(v): 3]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_17 [label=\"Variable_17\\n[l(v): 0, r(v): 3]\\na_1^(2)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_9 [label=\"Operator_9\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_8 [label=\"LogicalExpr_8\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_8 } -> { LogicalExpr_20, Operator_21, LogicalExpr_22 }\n"
      "\tLogicalExpr_22 [label=\"LogicalExpr_22\\n[l(v): 0, r(v): 2]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_22 } -> { Variable_23, Operator_24, Variable_25 }\n"
      "\tVariable_25 [label=\"Variable_25\\n[l(v): 0, r(v): 2]\\na_2^(1)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_24 [label=\"Operator_24\\n[l(v): 0, r(v): 2]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_23 [label=\"Variable_23\\n[l(v): 0, r(v): 2]\\na_2^(1)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_21 [label=\"Operator_21\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_20 [label=\"LogicalExpr_20\\n[l(v): 1, r(v): 2]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_20 } -> { Variable_26, Operator_27, Variable_28 }\n"
      "\tVariable_28 [label=\"Variable_28\\n[l(v): 0, r(v): 3]\\na_1^(1)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_27 [label=\"Operator_27\\n[l(v): 0, r(v): 3]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_26 [label=\"Variable_26\\n[l(v): 0, r(v): 3]\\na_1^(1)_left\" shape=oval style=filled fillcolor=white]\n"
      "}\n";

  ASSERT_EQ(outputStream.str(), expectedStr);
}

TEST_F(DotPrinterFixture, printAllReachableNods_printNodeSet) {  /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(18, ast);

  std::stringstream outputStream;

  DotPrinter dp;
  dp.setIndentationCharacter("\t")
      .setShowMultDepth(false)
      .setOutputStream(outputStream);

  dp.printAllReachableNodes(ast.getRootNode());

  auto expectedStr =
      "\tReturn_0 [label=\"Return_0\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Return_0 } -> { LogicalExpr_1 }\n"
      "\tLogicalExpr_1 [label=\"LogicalExpr_1\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_1 } -> { LogicalExpr_2, Operator_3, Variable_4 }\n"
      "\tLogicalExpr_2 [label=\"LogicalExpr_2\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_2 } -> { LogicalExpr_5, Operator_6, Variable_7 }\n"
      "\tOperator_3 [label=\"Operator_3\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_4 [label=\"Variable_4\\na_t\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_5 [label=\"LogicalExpr_5\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_5 } -> { LogicalExpr_8, Operator_9, LogicalExpr_10 }\n"
      "\tOperator_6 [label=\"Operator_6\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_7 [label=\"Variable_7\\ny_1\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_8 [label=\"LogicalExpr_8\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_8 } -> { Variable_11, Operator_12, Variable_13 }\n"
      "\tOperator_9 [label=\"Operator_9\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_10 [label=\"LogicalExpr_10\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_10 } -> { Variable_14, Operator_15, Variable_16 }\n"
      "\tVariable_11 [label=\"Variable_11\\na_1^(1)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_12 [label=\"Operator_12\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_13 [label=\"Variable_13\\na_2^(1)\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_14 [label=\"Variable_14\\na_1^(2)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_15 [label=\"Operator_15\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_16 [label=\"Variable_16\\na_2^(2)\" shape=oval style=filled fillcolor=white]\n";

  ASSERT_EQ(outputStream.str(), expectedStr);
}
