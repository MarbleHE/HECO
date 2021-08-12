#include <algorithm>
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
      n386 = a0  *** b0;
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

TEST(VerilogToDsl, testOpenFile) {

  VeriLogToDsl verilogParser("/Users/mwinger/Desktop/MasterThesis/ABC/cmake-build-debug/test/adder.v");

  verilogParser.parseFile();

}


TEST(VerilogToDsl, testInputParser) {

}


TEST(VerilogToDsl, testOutputParser) {

}


TEST(VerilogToDsl, testProgramParser) {

}

#endif