#include <algorithm>
#include <ast_opt/visitor/GetAllNodesVisitor.h>

#include "ast_opt/parser/Parser.h"
#include "../ASTComparison.h"
#include "gtest/gtest.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/utilities/ConeRewriter.h"

#ifdef HAVE_SEAL_BFV

TEST(ConeRewriterTest, testConeRewrNoChange) { /* NOLINT */

  /// ||(false,true,false,true,false,true,false,true,false,true)
  auto o =
      std::make_unique<OperatorExpression>(Operator(LOGICAL_OR), std::vector<std::unique_ptr<AbstractExpression>>());
  for (int i = 0; i < 10; ++i) {
    auto l = std::make_unique<LiteralBool>(i%2);
    o->appendOperand(std::move(l));
  }

//  std::stringstream ss;
//  PrintVisitor p(ss);
//  o->accept(p);
//  std::cout << ss.str() << std::endl;

  // Keep a copy of o for later comparison
  auto o_copy = o->clone();

  ConeRewriter coneRewriter;

  auto rewritten_ast = coneRewriter.rewriteAst(std::move(o));

  //In this case, asts should be identical
  ASSERT_NE(rewritten_ast, nullptr);
  compareAST(*o_copy, *rewritten_ast);
}


TEST(ConeRewriterTest, testConeRewrPaperTree) {
  // program's input
  const char *inputs = R""""(
      bool __input0__ = 0;
      bool __input1__ = 1;
      bool __input2__ = 0;
      bool __input3__ = 0;
      bool __input4__ = 1;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret bool v1 = __input0__ && __input1__;
      secret bool v2 = __input2__ && __input3__;
      secret bool y1 = v1 || v2;
      secret bool r  = y1 && __input4__;
      return r;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = r;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  std::stringstream ss;
  PrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;

  //coneRewriter.rewriteAst(astProgram.get());
}


TEST(ConeRewriterTest, testMultDepth) {

// program's input
  const char *inputs = R""""(
      bool __input0__ = 0;
      bool __input1__ = 1;
      bool __input2__ = 0;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret bool v1 = ((__input0__ && __input1__) && __input2__) && __input3__;

      return v2;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = v2;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  std::stringstream ss;
  PrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;

  auto depth = coneRewriter.getMultDepthL(astProgram.get());
  std::cout << depth << std::endl;
  ASSERT_EQ(depth, 3);
}
///
TEST(ConeRewriterTest, testReversedMultDepth) {

// program's input
  const char *inputs = R""""(
      bool __input0__ = 0;
      bool __input1__ = 1;
      bool __input2__ = 0;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret bool v1 = ((__input0__ && __input1__) && __input2__) && __input3__;

      return v2;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = v2;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  std::stringstream ss;
  PrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;

  auto depth = coneRewriter.getReverseMultDepthR(astProgram.get());
  std::cout << depth << std::endl;
  ASSERT_EQ(depth, 0);
}

TEST(ConeRewriterTest, testprecomputeMultDepths) {

// program's input
  const char *inputs = R""""(
      bool __input0__ = 0;
      bool __input1__ = 1;
      bool __input2__ = 0;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret bool v1 = ((__input0__ && __input1__) && __input2__) && __input3__;

      return v2;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = v2;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  std::stringstream ss;
  PrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;
  GetAllNodesVisitor vis;

  astProgram.get()->accept(vis);
  //vis.visit(*astProgram.get());
  std::cout << vis.v[0] << std::endl;

  coneRewriter.precomputeMultDepths(astProgram.get());

}

#endif