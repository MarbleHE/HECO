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

