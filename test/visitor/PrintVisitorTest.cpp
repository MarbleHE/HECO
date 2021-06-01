#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/visitor/NoisePrintVisitor.h"
#include "gtest/gtest.h"

TEST(PrintVisitor, printTree) {
  // Confirm that printing children works as expected

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  std::stringstream ss;
  PrintVisitor v(ss);
  v.visit(assignment);

  std::cout << ss.str();


  EXPECT_EQ(ss.str(),"NODE VISITED: Assignment\n"
                              "NODE VISITED:   Variable (foo)\n"
                              "LITERAL BOOL VISITED:   LiteralBool (true)\n");
}

