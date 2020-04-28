#include "Ast.h"
#include "PrintVisitor.h"
#include <fstream>
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"

class PrintVisitorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // this is necessary to ensure that node IDs (static variable) are always the same,
    // independent of which test run at first
    AbstractNode::resetNodeIdCounter();
  }
};

TEST_F(PrintVisitorTest, printDemoTreeOne) { /* NOLINT */
  Ast a;
  AstTestingGenerator::generateAst(14, a);
  PrintVisitor pv(false);
  pv.visit(a);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printDemoTreeOne.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printDemoTreeTwo) { /* NOLINT */
  Ast a;
  AstTestingGenerator::generateAst(15, a);
  PrintVisitor pv(false);
  pv.visit(a);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printDemoTreeTwo.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printAstIncludingForStatement) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(23, ast);
  PrintVisitor pv(false);
  pv.visit(ast);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printAstIncludingForStatement.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printAstUsingRotation) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(24, ast);
  PrintVisitor pv(false);
  pv.visit(ast);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printAstUsingRotation.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printRotationAst) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(26, ast);
  PrintVisitor pv(false);
  pv.visit(ast);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printRotationAst.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printGetMatrixElementAst) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(31, ast);
  PrintVisitor pv(false);
  pv.visit(ast);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printGetMatrixElementAst.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printSimpleMatrix) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(33, ast);
  PrintVisitor pv(false);
  pv.visit(ast);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printSimpleMatrix.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printOperatorExpr) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(35, ast);
  PrintVisitor pv(false);
  pv.visit(ast);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printOperatorExpr.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printMatrixAssignm) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(50, ast);
  PrintVisitor pv(false);
  pv.visit(ast);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printMatrixAssignm.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printMatrixPermutation) { /* NOLINT */
  Ast ast;
  AstTestingGenerator::generateAst(51, ast);
  PrintVisitor pv(false);
  pv.visit(ast);

  // read expected output file
  std::ifstream ifs("../../test/expected_output_large/PrintVisitorTest/printMatrixPermutation.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}
