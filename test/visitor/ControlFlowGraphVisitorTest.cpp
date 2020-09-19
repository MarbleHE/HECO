#include <include/ast_opt/utilities/Datatype.h>
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "include/ast_opt/visitor/ControlFlowGraph/ControlFlowGraphVisitor.h"
#include "gtest/gtest.h"

TEST(ControlFlowGraphVisitor, wip) {
  // Confirm that printing children works as expected

  // int scalar = {2};
  auto declarationScalar = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                                 std::move(std::make_unique<Variable>("scalar")),
                                                                 std::move(std::make_unique<LiteralInt>(2)));

  // int vec = {3, 4, 9, 2, 1};
  std::vector<std::unique_ptr<AbstractExpression>> exprs;
  exprs.emplace_back(std::make_unique<LiteralInt>(3));
  exprs.emplace_back(std::make_unique<LiteralInt>(4));
  exprs.emplace_back(std::make_unique<LiteralInt>(9));
  exprs.emplace_back(std::make_unique<LiteralInt>(2));
  exprs.emplace_back(std::make_unique<LiteralInt>(1));
  auto declarationVec = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                              std::make_unique<Variable>("vec"),
                                                              std::make_unique<ExpressionList>(std::move(exprs)));

  // public void main() { ... }
  std::vector<std::unique_ptr<AbstractStatement>> statements;
  statements.push_back(std::move(declarationScalar));
  statements.push_back(std::move(declarationVec));
  auto statementBlock = std::make_unique<Block>(std::move(statements));
  auto expected = std::make_unique<Function>(Datatype(Type::VOID),
                                             "main",
                                             std::move(std::vector<std::unique_ptr<FunctionParameter>>()),
                                             std::move(statementBlock));

//  std::stringstream ss;
  ControlFlowGraphVisitor cfgv;
  cfgv.visit(*expected);

  std::cout << "test" << std::endl;

//  EXPECT_EQ(ss.str(), "Assignment\n"
//                      "  Variable (foo)\n"
//                      "  LiteralBool (true)\n");
}
