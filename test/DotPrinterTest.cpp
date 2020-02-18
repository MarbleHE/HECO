#include "AbstractNode.h"
#include "Operator.h"
#include "Variable.h"
#include "BinaryExpr.h"
#include "Function.h"
#include "DotPrinter.h"
#include <fstream>
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"

class DotPrinterFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // This is required, otherwise the node IDs won't match the expected result as they are incremented ongoing but
    // must be reset after each test case.
    AbstractNode::resetNodeIdCounter();
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

  // read expected output from file
  std::ifstream
      ifs("../../test/expected_output_large/DotPrinterTest/printAsDotFormattedGraphTest_printAstExample1.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  ASSERT_EQ(buffer.str(), outputStream.str());
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

  // read expected output from file
  std::ifstream
      ifs("../../test/expected_output_large/DotPrinterTest/printAsDotFormattedGraphTest_printAstExample2.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  ASSERT_EQ(buffer.str(), outputStream.str());
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

  // read expected output from file
  std::ifstream ifs("../../test/expected_output_large/DotPrinterTest/printAllReachableNods_printNodeSet.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  ASSERT_EQ(buffer.str(), outputStream.str());
}
