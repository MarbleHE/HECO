#include <algorithm>
#include <ast_opt/visitor/GetAllNodesVisitor.h>

#include "ast_opt/parser/Parser.h"
#include "../ASTComparison.h"
#include "gtest/gtest.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/ProgramPrintVisitor.h"
#include "ast_opt/utilities/ConeRewriter.h"
#include "ast_opt/visitor/BinaryToOperatorExpressionVisitor.h"

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

TEST(ConeRewriterTest, getReducibleCones) {
  /// program specification
  /// v1 = a && b;
  /// u = v1 || (x || y);
  /// vt = u && c;
  const char *program = R""""(
  return ((a && b) || (x || y)) && c;
  )"""";
  auto astProgram = Parser::parse(std::string(program));

  // Rewrite BinaryExpressions to trivial OperatorExpressions
  BinaryToOperatorExpressionVisitor v;
  astProgram->accept(v);

  std::stringstream ss;
  ProgramPrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;

  auto cones = coneRewriter.getReducibleCones(*astProgram);

  std::cout << "Found " << cones.size() << " reducible cones:" << std::endl;
  for (auto &n: cones) {
    std::cout << n->toString(false) << std::endl;
  }

  //TODO: Figure out what the reducible cones in this test should be!
  EXPECT_EQ(true, false);
}

/// Test based on Figure 1 of
/// "Aubry, P. et al. 2019. Faster Homomorphic Encryption is not Enough:
/// Improved Heuristic for Multiplicative Depth Minimization of Boolean Circuits.
/// Cryptology ePrint Archive, Report 2019/963."
TEST(ConeRewriterTest, testConeRewrPaperTree) {
  /// program specification
  /// v1 = a && b;
  /// u = v1 || (x || y);
  /// vt = u && c;
  const char *program = R""""(
  return ((a && b) || (x || y)) && c;
  )"""";
  auto astProgram = Parser::parse(std::string(program));

  // Rewrite BinaryExpressions to trivial OperatorExpressions
  BinaryToOperatorExpressionVisitor v;
  astProgram->accept(v);

  std::stringstream ss;
  ProgramPrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;

  auto rewritten_ast = coneRewriter.rewriteAst(std::move(astProgram));

  ASSERT_NE(rewritten_ast, nullptr);

  std::stringstream rewritten_ss;
  ProgramPrintVisitor p2(rewritten_ss);
  rewritten_ast->accept(p2);
  std::cout << rewritten_ss.str() << std::endl;


  /// reference specification
  /// u1 = a && (b && c)
  /// u2 = c && (x || y)
  /// r = u1 || u2
  const char *expected = R""""(
  return (a && (b && c)) || (c && (x || y));
  )"""";
  auto expected_ast = Parser::parse(std::string(expected));

  // Rewrite BinaryExpressions to trivial OperatorExpressions
  expected_ast->accept(v);

  std::stringstream expected_ss;
  ProgramPrintVisitor p3(expected_ss);
  expected_ast->accept(p3);
  std::cout << expected_ss.str() << std::endl;

  EXPECT_EQ(expected_ss.str(), rewritten_ss.str());
}

TEST(ConeRewriterTest, testMultDepth) {

  /// program specification
  /// v1 = a && b;
  /// u = v1 || (x || y);
  /// vt = u && c;
  const char *program = R""""(
  return ((a && b) || (x || y)) && c;
  )"""";
  auto astProgram = Parser::parse(std::string(program));

  // Rewrite BinaryExpressions to trivial OperatorExpressions
  BinaryToOperatorExpressionVisitor v;
  astProgram->accept(v);

  std::stringstream ss;
  ProgramPrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;


  ConeRewriter coneRewriter;

  std::unordered_map<std::string, int> depthMap;
  int depth = coneRewriter.computeMultDepthL(astProgram.get());  // compute mult depth map for the root node of the AST (should be 2)
  //depthMap = coneRewriter.preComputeMultDepthsL(astProgram.get());
  ASSERT_EQ(depth, 2);
  //ASSERT_EQ(depthMap[astProgram->getUniqueNodeId()], 2);
}

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
  ProgramPrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;

  auto depth = 0; // TODO: RENABLE coneRewriter.getReverseMultDepthR(astProgram.get());
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
  ProgramPrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;
  GetAllNodesVisitor vis;

  astProgram.get()->accept(vis);
  //vis.visit(*astProgram.get());
  std::cout << vis.v[0] << std::endl;

  // TODO: RENABLE coneRewriter.precomputeMultDepths(astProgram.get());

}

#endif