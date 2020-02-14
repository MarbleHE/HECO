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
      "  BinaryExpr_2 [label=\"BinaryExpr_2\\n[l(v): 0, r(v): 0]\" shape=oval style=filled fillcolor=white]\n"
      "  { BinaryExpr_2 } -> { Variable_0, Operator_3, LiteralInt_1 }\n";

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
      "\tBinaryExpr_2 [label=\"BinaryExpr_2\\n[l(v): 0, r(v): 0]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ Variable_0, Operator_3, LiteralInt_1 } -> { BinaryExpr_2 }\n";

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
      "\tReturn_16 [label=\"Return_16\\n[l(v): 2, r(v): 0]\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Return_16 } -> { LogicalExpr_14 }\n"
      "\tLogicalExpr_14 [label=\"LogicalExpr_14\\n[l(v): 2, r(v): 0]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_14 } -> { LogicalExpr_11, Operator_15, Variable_13 }\n"
      "\tVariable_13 [label=\"Variable_13\\n[l(v): 0, r(v): 1]\\na_t\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_15 [label=\"Operator_15\\n[l(v): 0, r(v): 1]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_11 [label=\"LogicalExpr_11\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_11 } -> { LogicalExpr_8, Operator_12, Variable_10 }\n"
      "\tVariable_10 [label=\"Variable_10\\n[l(v): 0, r(v): 1]\\ny_1\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_12 [label=\"Operator_12\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_8 [label=\"LogicalExpr_8\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_8 } -> { LogicalExpr_2, Operator_9, LogicalExpr_6 }\n"
      "\tLogicalExpr_6 [label=\"LogicalExpr_6\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_6 } -> { Variable_4, Operator_7, Variable_5 }\n"
      "\tVariable_5 [label=\"Variable_5\\n[l(v): 0, r(v): 2]\\na_2^(2)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_7 [label=\"Operator_7\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_4 [label=\"Variable_4\\n[l(v): 0, r(v): 2]\\na_1^(2)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_9 [label=\"Operator_9\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_2 [label=\"LogicalExpr_2\\n[l(v): 1, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_2 } -> { Variable_0, Operator_3, Variable_1 }\n"
      "\tVariable_1 [label=\"Variable_1\\n[l(v): 0, r(v): 2]\\na_2^(1)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_3 [label=\"Operator_3\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
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
      "\tReturn_28 [label=\"Return_28\\n[l(v): 3, r(v): 0]\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Return_28 } -> { LogicalExpr_26 }\n"
      "\tLogicalExpr_26 [label=\"LogicalExpr_26\\n[l(v): 3, r(v): 0]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_26 } -> { LogicalExpr_23, Operator_27, Variable_25 }\n"
      "\tVariable_25 [label=\"Variable_25\\n[l(v): 0, r(v): 1]\\na_t\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_27 [label=\"Operator_27\\n[l(v): 0, r(v): 1]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_23 [label=\"LogicalExpr_23\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_23 } -> { LogicalExpr_20, Operator_24, Variable_22 }\n"
      "\tVariable_22 [label=\"Variable_22\\n[l(v): 0, r(v): 1]\\ny_1\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_24 [label=\"Operator_24\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_20 [label=\"LogicalExpr_20\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_20 } -> { LogicalExpr_8, Operator_21, LogicalExpr_18 }\n"
      "\tLogicalExpr_18 [label=\"LogicalExpr_18\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_18 } -> { LogicalExpr_12, Operator_19, LogicalExpr_16 }\n"
      "\tLogicalExpr_16 [label=\"LogicalExpr_16\\n[l(v): 0, r(v): 2]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_16 } -> { Variable_14, Operator_17, Variable_15 }\n"
      "\tVariable_15 [label=\"Variable_15\\n[l(v): 0, r(v): 2]\\na_2^(2)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_17 [label=\"Operator_17\\n[l(v): 0, r(v): 2]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_14 [label=\"Variable_14\\n[l(v): 0, r(v): 2]\\na_2^(2)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_19 [label=\"Operator_19\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_12 [label=\"LogicalExpr_12\\n[l(v): 1, r(v): 2]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_12 } -> { Variable_10, Operator_13, Variable_11 }\n"
      "\tVariable_11 [label=\"Variable_11\\n[l(v): 0, r(v): 3]\\na_1^(2)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_13 [label=\"Operator_13\\n[l(v): 0, r(v): 3]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_10 [label=\"Variable_10\\n[l(v): 0, r(v): 3]\\na_1^(2)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_21 [label=\"Operator_21\\n[l(v): 0, r(v): 1]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_8 [label=\"LogicalExpr_8\\n[l(v): 2, r(v): 1]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_8 } -> { LogicalExpr_2, Operator_9, LogicalExpr_6 }\n"
      "\tLogicalExpr_6 [label=\"LogicalExpr_6\\n[l(v): 0, r(v): 2]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_6 } -> { Variable_4, Operator_7, Variable_5 }\n"
      "\tVariable_5 [label=\"Variable_5\\n[l(v): 0, r(v): 2]\\na_2^(1)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_7 [label=\"Operator_7\\n[l(v): 0, r(v): 2]\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_4 [label=\"Variable_4\\n[l(v): 0, r(v): 2]\\na_2^(1)_left\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_9 [label=\"Operator_9\\n[l(v): 0, r(v): 2]\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_2 [label=\"LogicalExpr_2\\n[l(v): 1, r(v): 2]\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_2 } -> { Variable_0, Operator_3, Variable_1 }\n"
      "\tVariable_1 [label=\"Variable_1\\n[l(v): 0, r(v): 3]\\na_1^(1)_right\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_3 [label=\"Operator_3\\n[l(v): 0, r(v): 3]\\nAND\" shape=oval style=filled fillcolor=white]\n"
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
      "\tReturn_16 [label=\"Return_16\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Return_16 } -> { LogicalExpr_14 }\n"
      "\tLogicalExpr_14 [label=\"LogicalExpr_14\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_14 } -> { LogicalExpr_11, Operator_15, Variable_13 }\n"
      "\tLogicalExpr_11 [label=\"LogicalExpr_11\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_11 } -> { LogicalExpr_8, Operator_12, Variable_10 }\n"
      "\tOperator_15 [label=\"Operator_15\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_13 [label=\"Variable_13\\na_t\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_8 [label=\"LogicalExpr_8\" shape=oval style=filled fillcolor=white]\n"
      "\t{ LogicalExpr_8 } -> { LogicalExpr_2, Operator_9, LogicalExpr_6 }\n"
      "\tOperator_12 [label=\"Operator_12\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_10 [label=\"Variable_10\\ny_1\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_2 [label=\"LogicalExpr_2\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_2 } -> { Variable_0, Operator_3, Variable_1 }\n"
      "\tOperator_9 [label=\"Operator_9\\nXOR\" shape=oval style=filled fillcolor=white]\n"
      "\tLogicalExpr_6 [label=\"LogicalExpr_6\" shape=oval style=filled fillcolor=red]\n"
      "\t{ LogicalExpr_6 } -> { Variable_4, Operator_7, Variable_5 }\n"
      "\tVariable_0 [label=\"Variable_0\\na_1^(1)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_3 [label=\"Operator_3\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_1 [label=\"Variable_1\\na_2^(1)\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_4 [label=\"Variable_4\\na_1^(2)\" shape=oval style=filled fillcolor=white]\n"
      "\tOperator_7 [label=\"Operator_7\\nAND\" shape=oval style=filled fillcolor=white]\n"
      "\tVariable_5 [label=\"Variable_5\\na_2^(2)\" shape=oval style=filled fillcolor=white]\n";

  ASSERT_EQ(outputStream.str(), expectedStr);
}
