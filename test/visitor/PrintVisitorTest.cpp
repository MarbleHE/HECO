#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableAssignment.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "gtest/gtest.h"

TEST(PrintVisitor, printTree) {
  // Confirm that printing children works as expected

  VariableAssignment assignment(std::make_unique<Variable>("foo"),std::make_unique<LiteralBool>(true));

  std::stringstream ss;
  PrintVisitor v(ss);
  v.visit(assignment);

  EXPECT_EQ(ss.str(),"VariableAssignment\n"
                              "  Variable (foo)\n"
                              "  LiteralBool (true)\n");
}

//TODO: Extend to non-trivial trees