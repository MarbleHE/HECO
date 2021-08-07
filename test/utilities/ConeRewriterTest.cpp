#include <algorithm>
#include <ast_opt/visitor/GetAllNodesVisitor.h>
#include "ast_opt/parser/Parser.h"
#include "../ASTComparison.h"
#include "gtest/gtest.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/ProgramPrintVisitor.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/utilities/ConeRewriter.h"
#include "ast_opt/visitor/BinaryToOperatorExpressionVisitor.h"
#include "ast_opt/visitor/ParentSettingVisitor.h"

#ifdef HAVE_SEAL_BFV


/// Test based on Figure 1 of
/// "Aubry, P. et al. 2019. Faster Homomorphic Encryption is not Enough:
/// Improved Heuristic for Multiplicative Depth Minimization of Boolean Circuits.
/// Cryptology ePrint Archive, Report 2019/963.
/// a    b  x   y
///  \  /    \ /
///   AND     OR
///     \    /
///       OR    c
///         \  /
///          AND
///           |
///           r


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

TEST(ConeRewriterTest, getReducibleConePaperCircuit) {
  /// Expected output of test (node marked by (*))
  ///  a    b  x   y
  ///   \  /    \ /
  ///   AND     OR
  ///     \    /
  ///      OR    c
  ///       \  /
  ///       AND(*)
  ///         |
  ///         r

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
  PrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  auto cones = ConeRewriter::getReducibleCone(astProgram.get(), astProgram.get(), 1, {}); //TODO: check if this is actually the node v from the paper (last AND)

  std::cout << "Found " << cones.size() << " reducible cones:" << std::endl;
  for (auto &n: cones) {
    std::cout << n->toString(false) << std::endl;
  }
  ASSERT_EQ(cones.size(), 1);

  EXPECT_EQ(cones[0]->getUniqueNodeId(), astProgram->begin()->begin()->getUniqueNodeId());
}

TEST(ConeRewriterTest, getReducibleConeMoreInterestingCircuit) {
  /// Expected output of test (node marked by (*))
  ///    a1 a2                 a8 a9
  ///    \ /                    \ /
  ///    AND    a3      a6  a7  AND   a10      a13  a14
  ///      \   /         \ /     \   /         \   /
  ///       AND  a4  a5  AND     AND  a11  a12  AND
  ///        \  /    \  /         \  /     \   /
  ///        AND    AND           AND       AND
  ///         \     /              \       /
  ///          \   /                \     /
  ///          AND                    AND
  ///           \                    /
  ///            \    y1  y2        /
  ///             \  /      \     /
  ///             OR (~)    OR (~)
  ///              \      /
  ///               AND (*)
  ///                |
  ///                r
  ///
  /// expected output nodes (*)  and one of the nodes (~) (randomly selected by the algo)


  const char *program = R""""(
  return  ((((a1 && a2 ) && a3 ) && a4) && (a5 && (a6 && a7)) || y1) && ((((a8 && a9 ) && a10 ) && a11) && (a12 && (a13 && a14)) || y2);
  )"""";
  auto astProgram = Parser::parse(std::string(program));

  // Rewrite BinaryExpressions to trivial OperatorExpressions
  BinaryToOperatorExpressionVisitor v;
  astProgram->accept(v);

  std::stringstream ss;
  PrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;


  std::cout << "NOdee: " << astProgram->begin()->begin()->begin()->toString(false) << " " << astProgram->begin()->begin()->begin()->getUniqueNodeId() << std::endl;



  auto cones = ConeRewriter::getReducibleCone(astProgram.get(), astProgram.get(), 1, {});

  std::cout << "Found " << cones.size() << " reducible cones:" << std::endl;
  for (auto &n: cones) {
    std::cout << n->toString(false) << std::endl;
  }
  ASSERT_EQ(cones.size(), 2);

  EXPECT_TRUE(((cones[0]->getUniqueNodeId() == astProgram->begin()->begin()->getUniqueNodeId()) && (cones[1]->getUniqueNodeId() ==  astProgram->begin()->begin()->begin()->getUniqueNodeId()))
                    || ((cones[0]->getUniqueNodeId() == astProgram->begin()->begin()->getUniqueNodeId()) &&  (cones[1]->getUniqueNodeId() ==  astProgram->begin()->begin()->end()->getUniqueNodeId())) );

}

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
  /// excpected mult depths:
  /// a    b  x   y
  ///  \  /    \ /
  ///    1      0
  ///     \   /
  ///       1    c
  ///        \  /
  ///          2
  ///          |
  ///          r

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

  MultDepthMap depthMap;
  int depth = coneRewriter.computeMultDepthL(astProgram.get(),
                                             depthMap);  // compute mult depth map for the root node of the AST (should be 2)
  //depthMap = coneRewriter.preComputeMultDepthsL(astProgram.get());
  ASSERT_EQ(depth, 2);
  //ASSERT_EQ(depthMap[astProgram->getUniqueNodeId()], 2);
}

TEST(ConeRewriterTest, testReversedMultDepth) {
  /// excpected reversed mult depths:
  /// a    b  x   y
  ///  \  /    \ /
  ///    1      1
  ///     \   /
  ///       1    c
  ///        \  /
  ///          0
  ///          |
  ///          r

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

  MultDepthMap depthMap;
  int depth = coneRewriter.computeReversedMultDepthR(astProgram
                                                         .get(),
                                                     depthMap,
                                                     nullptr);  // compute mult depth map for the root node of the AST (should be 2)
  //depthMap = coneRewriter.preComputeMultDepthsL(astProgram.get());
  ASSERT_EQ(depth, 0);
}

TEST(ConeRewriterTest, testComputeAllDepths) {

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

  // Get nodes, but only expression nodes, not the block or return
  GetAllNodesVisitor vis;
  astProgram->begin()->begin()->accept(vis);

  ConeRewriter coneRewriter;
  MultDepthMap multDepths;
  for (auto n : vis.v) {
    coneRewriter.computeMultDepthL(n, multDepths);
  }

  for (auto n : vis.v) {
    std::cout << "Node: " << n->toString(false) << "Id: " << n->getUniqueNodeId() << " MultDepth: " << multDepths[n->getUniqueNodeId()] << std::endl;
  }

  EXPECT_EQ(multDepths["OperatorExpression_19"], 2);
  EXPECT_EQ(multDepths["OperatorExpression_18"], 1);
  EXPECT_EQ(multDepths["OperatorExpression_16"], 1);
  EXPECT_EQ(multDepths["OperatorExpression_17"], 0);
  EXPECT_EQ(multDepths["Variable_2"], 0);
  EXPECT_EQ(multDepths["Variable_4"], 0);
  EXPECT_EQ(multDepths["Variable_7"], 0);
  EXPECT_EQ(multDepths["Variable_9"], 0);
  EXPECT_EQ(multDepths["Variable_13"], 0);
}

TEST(ConeRewriterTest, testComputeAllReversedMultDepthsR) {

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
  PrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;

  GetAllNodesVisitor vis;
  astProgram->accept(vis);


  MultDepthMap revMultDepths;
//  for (auto n : vis.v) {
//    coneRewriter.computeReversedMultDepthR(n, revMultDepths);
//  }

//TODO FIX BinToOPvisitor, then this will work!

//std::cout << "Testing on node " <<
//                                 astProgram->begin()->begin()->begin()->begin()->begin()->getUniqueNodeId() << " which is " <<
//                                 astProgram->begin()->begin()->begin()->begin()->begin()->toString(false) << std::endl;
//std::cout << "Its parent is " << astProgram->begin()->begin()->begin()->begin()->begin()->getParent().getUniqueNodeId() <<
//             " which is " <<     astProgram->begin()->begin()->begin()->begin()->begin()->getParent().toString(false) << std::endl;
//std::cout << "Result: " << coneRewriter.computeReversedMultDepthR(&*astProgram->begin()->begin()->begin()->begin()->begin(), revMultDepths, nullptr) << std::endl;

  for (auto n : vis.v) {
    std::cout << "Node: " << n->toString(false) << "Id: " << n->getUniqueNodeId() << " Rev: " << coneRewriter.computeReversedMultDepthR(n, revMultDepths);
  }

  EXPECT_EQ(revMultDepths["OperatorExpression_19"], 0);
  EXPECT_EQ(revMultDepths["OperatorExpression_18"], 1);
  EXPECT_EQ(revMultDepths["OperatorExpression_16"], 1);
  EXPECT_EQ(revMultDepths["OperatorExpression_17"], 1);
  EXPECT_EQ(revMultDepths["Variable_2"], 2);
  EXPECT_EQ(revMultDepths["Variable_4"], 2);
  EXPECT_EQ(revMultDepths["Variable_7"], 1);
  EXPECT_EQ(revMultDepths["Variable_9"], 1);
  EXPECT_EQ(revMultDepths["Variable_13"], 1);
}

TEST(ConeRewriterTest, computeMinDepthTest) {
  /// a    b  x   y
  ///  \  /    \ /
  ///   AND     OR
  ///     \    /
  ///       OR    c
  ///         \  /
  ///          1
  ///           |
  ///           r

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
  MultDepthMap map;

  int minDepth = coneRewriter.computeMinDepth(&*astProgram->begin()->begin(), astProgram.get(), map);

  ASSERT_EQ(minDepth, 1);
}

TEST(ConeRewriterTest, isCriticalNodeTest) {

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

  bool isCriticalNode = coneRewriter.isCriticalNode(astProgram.get(), astProgram.get());

  EXPECT_EQ(false, isCriticalNode);

}

TEST(ConeRewriterTest, getMaximumMultDepthTest) {
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

  int maximumMultDepth = coneRewriter.getMaximumMultDepth(astProgram.get());

  EXPECT_EQ(2, maximumMultDepth);
}

TEST(ConeRewriterTest, getAndCriticalCircuitTest) {
  /// expected graph:
  /// AND
  ///   \
  ///   AND
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
  PrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  ConeRewriter coneRewriter;

  //TODO we want a vector describing the grapth. only AND nodes
}

#endif