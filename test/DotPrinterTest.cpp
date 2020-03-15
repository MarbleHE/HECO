#include "DotPrinter.h"
#include "AbstractNode.h"
#include "AstTestingGenerator.h"
#include "ArithmeticExpr.h"
#include "Function.h"
#include "Operator.h"
#include "Variable.h"
#include "gtest/gtest.h"
#include <fstream>
#include "Block.h"

class DotPrinterFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // This is required, otherwise the node IDs won't match the expected result
    // as they are incremented ongoing but must be reset after each test case.
    AbstractNode::resetNodeIdCounter();
  }
};

TEST_F(DotPrinterFixture, getDotFormattedStringTest_printSimpleArithmeticExpression) { /* NOLINT */
  auto arithmeticExpression = new ArithmeticExpr(
      new Variable("alpha"), ArithmeticOp::MULTIPLICATION, new LiteralInt(212));

  auto expectedStr =
      "  ArithmeticExpr_3 [label=\"ArithmeticExpr_3\\n[l(v): 0, r(v): 0]\" shape=oval style=filled fillcolor=white]\n"
      "  { ArithmeticExpr_3 } -> { Variable_0, Operator_4, LiteralInt_2 }\n";

  Ast ast(arithmeticExpression);
  MultiplicativeDepthCalculator mdc(ast);
  DotPrinter dp;
  dp.setIndentationCharacter("  ")
      .setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(true);

  ASSERT_EQ(dp.getDotFormattedString(arithmeticExpression), expectedStr);
}

TEST_F(DotPrinterFixture, getDotFormattedStringTest_printReversedArithmeticExpression) { /* NOLINT */
  auto arithmeticExpression = new ArithmeticExpr(
      new Variable("alpha"), ArithmeticOp::MULTIPLICATION, new LiteralInt(212));

  // reversing the edge should only flip parents with children
  arithmeticExpression->swapChildrenParents();

  auto expectedStr =
      "\tArithmeticExpr_3 [label=\"ArithmeticExpr_3\\n[l(v): 0, r(v): 0]\" shape=oval style=filled fillcolor=white]\n"
      "\t{ Variable_0, Operator_4, LiteralInt_2 } -> { ArithmeticExpr_3 }\n";

  Ast ast(arithmeticExpression);
  MultiplicativeDepthCalculator mdc(ast);
  DotPrinter dp;
  dp.setIndentationCharacter("\t")
      .setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(true);

  ASSERT_EQ(dp.getDotFormattedString(arithmeticExpression), expectedStr);
}

TEST_F(DotPrinterFixture, getDotFormattedStringTest_printFunction) { /* NOLINT */
  auto functionParameters =
      new ParameterList({new FunctionParameter(new Datatype(Types::INT),
                                               new Variable("numberA")),
                         new FunctionParameter(new Datatype(Types::INT),
                                               new Variable("numberB"))});
  auto functionStatements = new Block(new VarDecl("numberC", 152));
  auto function = new Function("computeAverage", functionParameters, functionStatements);

  // this is needed to enable printing of the multiplicative depth
  Ast ast(function);
  MultiplicativeDepthCalculator mdc(ast);

  // create and configure the DotPrinter
  DotPrinter dp;
  dp.setIndentationCharacter("\t")
      .setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(true);

  auto expectedStr =
      "\tFunction_12 [label=\"Function_12\\n[l(v): 0, r(v): 0]\" shape=rect style=filled fillcolor=white]\n"
      "\t{ Function_12 } -> { ParameterList_6, Block_11 }\n";

  // check that Function is printed properly
  ASSERT_EQ(dp.getDotFormattedString(function), expectedStr);
}

TEST_F(DotPrinterFixture, printAsDotFormattedGraphTest_printCircuitExample1) { /* NOLINT */
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
  std::ifstream ifs("../../test/expected_output_large/DotPrinterTest/"
                    "printAsDotFormattedGraphTest_printCircuitExample1.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  ASSERT_EQ(buffer.str(), outputStream.str());
}

TEST_F(DotPrinterFixture, printAsDotFormattedGraphTest_printCircuitExample2) { /* NOLINT */
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
  std::ifstream ifs("../../test/expected_output_large/DotPrinterTest/"
                    "printAsDotFormattedGraphTest_printCircuitExample2.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  ASSERT_EQ(buffer.str(), outputStream.str());
}

TEST_F(DotPrinterFixture, printAsDotFormattedGraphTest_printAstExample1) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(17, ast);

  MultiplicativeDepthCalculator mdc(ast);
  std::stringstream outputStream;
  DotPrinter dp;
  dp.setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(true)
      .setOutputStream(outputStream);
  dp.printAsDotFormattedGraph(ast);

  // read expected output from file
  std::ifstream ifs("../../test/expected_output_large/DotPrinterTest/"
                    "printAsDotFormattedGraphTest_printAstExample1.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  ASSERT_EQ(buffer.str(), outputStream.str());
}

TEST_F(DotPrinterFixture, printAllReachableNods_printNodeSet) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(18, ast);

  std::stringstream outputStream;

  DotPrinter dp;
  dp.setIndentationCharacter("\t").setShowMultDepth(false).setOutputStream(
      outputStream);

  dp.printAllReachableNodes(ast.getRootNode());

  // read expected output from file
  std::ifstream ifs("../../test/expected_output_large/DotPrinterTest/"
                    "printAllReachableNods_printNodeSet.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  ASSERT_EQ(buffer.str(), outputStream.str());
}

TEST_F(DotPrinterFixture, printAsDotFormattedGraphTest_printAstIncludingIfStatement_withoutMultDepth) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(23, ast);

  MultiplicativeDepthCalculator mdc(ast);
  std::stringstream outputStream;
  DotPrinter dp;
  dp.setMultiplicativeDepthsCalculator(mdc)
      .setShowMultDepth(false)
      .setOutputStream(outputStream);
  dp.printAsDotFormattedGraph(ast);

  // read expected output from file
  std::ifstream ifs("../../test/expected_output_large/DotPrinterTest/"
                    "printAsDotFormattedGraphTest_printAstIncludingForStatement_withoutMultDepth.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  ASSERT_EQ(buffer.str(), outputStream.str());
}
