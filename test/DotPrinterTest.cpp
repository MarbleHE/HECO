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
};

TEST_F(DotPrinterFixture, getDotFormattedStringTest_printSimpleBinaryExpression) {  /* NOLINT */
  auto binaryExpression = new BinaryExpr(
      new Variable("alpha"),
      OpSymb::multiplication,
      new LiteralInt(212));

  auto expectedStr =
      "  BinaryExpr_3 [label=\"BinaryExpr_3\\n[l(v): 0, r(v): 0]\" shape=oval style=filled fillcolor=white]\n"
      "  { BinaryExpr_3 } -> { Variable_0, Operator_1, LiteralInt_2 }\n";

  Ast ast(binaryExpression);
  MultiplicativeDepthCalculator mdc(ast);
  DotPrinter dp;
  dp.setIndentationCharacter("  ")
      .setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(true);

  ASSERT_EQ(dp.getDotFormattedString(binaryExpression), expectedStr);
}

TEST_F(DotPrinterFixture, getDotFormattedStringTest_printReversedBinaryExpression) {  /* NOLINT */
  auto binaryExpression = new BinaryExpr(
      new Variable("alpha"),
      OpSymb::multiplication,
      new LiteralInt(212));

  // reversing the edge should only flip parents with children
  binaryExpression->swapChildrenParents();

  auto expectedStr =
      "\tBinaryExpr_3 [label=\"BinaryExpr_3\\n[l(v): 0, r(v): 0]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ Variable_0, Operator_1, LiteralInt_2 } -> { BinaryExpr_3 }\n";

  Ast ast(binaryExpression);
  MultiplicativeDepthCalculator mdc(ast);
  DotPrinter dp;
  dp.setIndentationCharacter("\t")
      .setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(true);

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

  MultiplicativeDepthCalculator mdc(ast);
  std::stringstream outputStream;
  DotPrinter dp;
  dp.setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(true)
      .setOutputStream(outputStream);
  dp.printAsDotFormattedGraph(ast);

  auto expectedStr =
      "digraph D {\n"
      "\tReturn_15 [label=\"Return_15\\n[l(v): 2, r(v): 0]\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Return_15 } -> { LogicalExpr_16 }\n"
      "\tLogicalExpr_16 [label=\"LogicalExpr_16\\n[l(v): 2, r(v): 0]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_16 } -> { LogicalExpr_12, Operator_13, Variable_14 }\n"
      "\tVariable_14 [label=\"Variable_14\\n[l(v): 0, r(v): 1]\\na_t\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_13 [label=\"Operator_13\\n[l(v): 0, r(v): 1]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_12 [label=\"LogicalExpr_12\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_12 } -> { LogicalExpr_9, Operator_10, Variable_11 }\n"
      "\tVariable_11 [label=\"Variable_11\\n[l(v): 0, r(v): 1]\\ny_1\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_10 [label=\"Operator_10\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_9 [label=\"LogicalExpr_9\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_9 } -> { LogicalExpr_3, Operator_4, LogicalExpr_5 }\n"
      "\tLogicalExpr_5 [label=\"LogicalExpr_5\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_5 } -> { Variable_6, Operator_7, Variable_8 }\n"
      "\tVariable_8 [label=\"Variable_8\\n[l(v): 0, r(v): 2]\\na_2^(2)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_7 [label=\"Operator_7\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_6 [label=\"Variable_6\\n[l(v): 0, r(v): 2]\\na_1^(2)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_4 [label=\"Operator_4\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_3 [label=\"LogicalExpr_3\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_3 } -> { Variable_0, Operator_1, Variable_2 }\n"
      "\tVariable_2 [label=\"Variable_2\\n[l(v): 0, r(v): 2]\\na_2^(1)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_1 [label=\"Operator_1\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_0 [label=\"Variable_0\\n[l(v): 0, r(v): 2]\\na_1^(1)\" shape=oval style=filled fillcolor=white]\n"
      "}\n";

  ASSERT_EQ(outputStream.str(), expectedStr);
}

TEST_F(DotPrinterFixture, printAsDotFormattedGraphTest_printAstExample2) {  /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(19, ast);

  MultiplicativeDepthCalculator mdc(ast);
  std::stringstream outputStream;
  DotPrinter dp;
  dp.setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(true)
      .setOutputStream(outputStream);
  dp.printAsDotFormattedGraph(ast);

  auto expectedStr =
      "digraph D {\n"
      "\tReturn_27 [label=\"Return_27\\n[l(v): 3, r(v): 0]\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Return_27 } -> { LogicalExpr_28 }\n"
      "\tLogicalExpr_28 [label=\"LogicalExpr_28\\n[l(v): 3, r(v): 0]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_28 } -> { LogicalExpr_24, Operator_25, Variable_26 }\n"
      "\tVariable_26 [label=\"Variable_26\\n[l(v): 0, r(v): 1]\\na_t\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_25 [label=\"Operator_25\\n[l(v): 0, r(v): 1]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_24 [label=\"LogicalExpr_24\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_24 } -> { LogicalExpr_21, Operator_22, Variable_23 }\n"
      "\tVariable_23 [label=\"Variable_23\\n[l(v): 0, r(v): 1]\\ny_1\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_22 [label=\"Operator_22\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_21 [label=\"LogicalExpr_21\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_21 } -> { LogicalExpr_9, Operator_10, LogicalExpr_11 }\n"
      "\tLogicalExpr_11 [label=\"LogicalExpr_11\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_11 } -> { LogicalExpr_12, Operator_16, LogicalExpr_17 }\n"
      "\tLogicalExpr_17 [label=\"LogicalExpr_17\\n[l(v): 0, r(v): 2]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_17 } -> { Variable_18, Operator_19, Variable_20 }\n"
      "\tVariable_20 [label=\"Variable_20\\n[l(v): 0, r(v): 2]\\na_2^(2)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_19 [label=\"Operator_19\\n[l(v): 0, r(v): 2]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_18 [label=\"Variable_18\\n[l(v): 0, r(v): 2]\\na_2^(2)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_16 [label=\"Operator_16\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_12 [label=\"LogicalExpr_12\\n[l(v): 1, r(v): 2]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_12 } -> { Variable_13, Operator_14, Variable_15 }\n"
      "\tVariable_15 [label=\"Variable_15\\n[l(v): 0, r(v): 3]\\na_1^(2)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_14 [label=\"Operator_14\\n[l(v): 0, r(v): 3]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_13 [label=\"Variable_13\\n[l(v): 0, r(v): 3]\\na_1^(2)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_10 [label=\"Operator_10\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_9 [label=\"LogicalExpr_9\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_9 } -> { LogicalExpr_3, Operator_4, LogicalExpr_5 }\n"
      "\tLogicalExpr_5 [label=\"LogicalExpr_5\\n[l(v): 0, r(v): 2]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_5 } -> { Variable_6, Operator_7, Variable_8 }\n"
      "\tVariable_8 [label=\"Variable_8\\n[l(v): 0, r(v): 2]\\na_2^(1)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_7 [label=\"Operator_7\\n[l(v): 0, r(v): 2]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_6 [label=\"Variable_6\\n[l(v): 0, r(v): 2]\\na_2^(1)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_4 [label=\"Operator_4\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_3 [label=\"LogicalExpr_3\\n[l(v): 1, r(v): 2]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_3 } -> { Variable_0, Operator_1, Variable_2 }\n"
      "\tVariable_2 [label=\"Variable_2\\n[l(v): 0, r(v): 3]\\na_1^(1)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_1 [label=\"Operator_1\\n[l(v): 0, r(v): 3]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_0 [label=\"Variable_0\\n[l(v): 0, r(v): 3]\\na_1^(1)_left\" shape=oval style=filled fillcolor=white]\n"
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
