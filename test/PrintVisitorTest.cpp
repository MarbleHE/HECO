#include <Ast.h>
#include <include/visitor/PrintVisitor.h>
#include <fstream>
#include "gtest/gtest.h"
#include "examples/genAstDemo.cpp"

class PrintVisitorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // this is necessary to ensure that node IDs (static variable) are the same, independent of which test run
    // at first
    Node::resetNodeIdCounter();
  }
};

TEST_F(PrintVisitorTest, printDemoTreeTwo) { /* NOLINT */
  Ast a;
  generateDemoTwo(a);
  PrintVisitor pv(false);
  pv.visit(a);

  // read expected output file
  std::ifstream ifs("../../test/aux/PrintVisitorTest/printDemoTreeTwo.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}

TEST_F(PrintVisitorTest, printDemoTreeOne) { /* NOLINT */
  Ast a;
  generateDemoOne(a);
  PrintVisitor pv(false);
  pv.visit(a);

  // read expected output file
  std::ifstream ifs("../../test/aux/PrintVisitorTest/printDemoTreeOne.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();

  EXPECT_EQ(pv.getOutput(), buffer.str());
}
