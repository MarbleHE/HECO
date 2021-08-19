#include <algorithm>
#include <vector>
#include <ast_opt/visitor/GetAllNodesVisitor.h>
#include "ast_opt/parser/Parser.h"
#include "../ASTComparison.h"
#include "gtest/gtest.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/ProgramPrintVisitor.h"
#include "ast_opt/utilities/VeriLogToDsl.h"


#ifdef HAVE_SEAL_BFV



TEST(VerilogToDsl, testPrelim) {

/// Test what input should look like:
/// ...
///

  // program's input
  const char *inputs = R""""(
      secret int a0;
      secret int b0;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int n386 = a0  *** b0;
      secret int n387 = (1 - a0) *** b0;
      secret int cOut = n1399 +++ n1404 --- n1399 *** n1404;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = n386;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  std::stringstream rr;
  ProgramPrintVisitor p(rr);
  astProgram->accept(p);
  std::cout << rr.str() << std::endl;
}

TEST(VerilogToDsl, testTokenizeFile) {
  VeriLogToDsl verilogParser("adder.v");
  verilogParser.tokenizeFile();
  EXPECT_EQ( verilogParser.getTokens().size(), 11959);
}

TEST(VerilogToDsl, testInputParser) {
  VeriLogToDsl verilogParser("test.v");
  verilogParser.tokenizeFile();
  verilogParser.parseInput();
  std::vector<std::string> inputVector = verilogParser.getInputs();
  EXPECT_EQ(inputVector.size(), 2);
  EXPECT_EQ(inputVector[0], "a0");
  EXPECT_EQ(inputVector[1], "a1");
}

TEST(VerilogToDsl, testOutputParser) {
  VeriLogToDsl verilogParser("test.v");
  verilogParser.tokenizeFile();
  verilogParser.parseOutput();
  std::vector<std::string> outputVector = verilogParser.getOutputs();
  EXPECT_EQ(outputVector.size(), 4);
  EXPECT_EQ(outputVector[0], "f0");
  EXPECT_EQ(outputVector[1], "f1");
  EXPECT_EQ(outputVector[2], "f2");
  EXPECT_EQ(outputVector[3], "cOut");
}

TEST(VerilogToDsl, testFindAssignBlocks) {
  VeriLogToDsl verilogParser("test.v");
  verilogParser.tokenizeFile();
  int firstAssign = verilogParser.findNextAssignmentBlock(0);
  int secondAssign = verilogParser.findNextAssignmentBlock(firstAssign);

  EXPECT_EQ(firstAssign, 19);
  EXPECT_EQ(secondAssign, 29);

}

TEST(VerilogToDsl, testSingleAssignmentParser) {
  /// checks if a single assignment is parsed correctly
  VeriLogToDsl verilogParser("test.v");
  verilogParser.tokenizeFile();
  int firstAssign = verilogParser.findNextAssignmentBlock(0);
  int secondAssign = verilogParser.findNextAssignmentBlock(firstAssign);
  std::vector<std::string> assignment = verilogParser.parseSingleAssignment(firstAssign, secondAssign);

  EXPECT_EQ("n386", assignment[0]);
  EXPECT_EQ("a0", assignment[1]);
  EXPECT_EQ("&", assignment[2]);
  EXPECT_EQ("~b0", assignment[3]);

}

TEST(VerilogToDsl, testAllAssignmentsParser) {
  /// checks if all assignments are parsed correctly
  VeriLogToDsl verilogParser("test.v");
  verilogParser.tokenizeFile();
  verilogParser.parseAllAssignments();
  std::vector<std::vector<std::string>> assignments = verilogParser.getAssignments();

  EXPECT_EQ(assignments.size(), 3);
  EXPECT_EQ("n386", assignments[0][0]);
  EXPECT_EQ("a0", assignments[0][1]);
  EXPECT_EQ("&", assignments[0][2]);
  EXPECT_EQ("~b0", assignments[0][3]);
}

TEST(VerilogToDsl, testPrintInputAsDSL) {
  VeriLogToDsl verilogParser("max.v");
  verilogParser.tokenizeFile();
  verilogParser.parseInput();
  verilogParser.writeInputToDsl("input_max.dsl");

}

TEST(VerilogToDsl, testPrintAssignmentsAsDSL) {
  VeriLogToDsl verilogParser("max.v");
  verilogParser.tokenizeFile();
  verilogParser.parseAllAssignments();
  verilogParser.writeAssignmentsToDsl("assignment_max.dsl");
}

#endif