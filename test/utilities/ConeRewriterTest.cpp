#include <algorithm>

#include "../ASTComparison.h"
#include "gtest/gtest.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/utilities/ConeRewriter.h"


#ifdef HAVE_SEAL_BFV

TEST(ConeRewriterTest, testConeRewr) { /* NOLINT */
  auto o =
      std::make_unique<OperatorExpression>(Operator(LOGICAL_OR), std::vector<std::unique_ptr<AbstractExpression>>());
  for (int i = 0; i < 10; ++i) {
    auto l = std::make_unique<LiteralBool>(i%2);
    o->appendOperand(std::move(l));
  }

  std::stringstream ss;
  PrintVisitor p(ss);
  o->accept(p);
  std::cout << ss.str() << std::endl;

  // Keep a copy of o for later comparison
  auto o_copy = o->clone();

  ConeRewriter coneRewriter;

  auto rewritten_ast = coneRewriter.rewriteAst(std::move(o));

  //In this case, asts should be identical
  ASSERT_NE(rewritten_ast, nullptr);
  compareAST(*o_copy, *rewritten_ast);
}

//TODO: Write tests where something should actually be rewritten (e.g. examples from paper)
#endif