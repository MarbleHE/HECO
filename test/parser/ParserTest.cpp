#include <typeinfo>

#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/parser/Parser.h"
#include "gtest/gtest.h"
#include "ParserTestHelpers.h"

Datatype BOOL = Datatype(Type::BOOL);
Datatype CHAR = Datatype(Type::CHAR);
Datatype INT = Datatype(Type::INT);
Datatype FLOAT = Datatype(Type::FLOAT);
Datatype DOUBLE = Datatype(Type::DOUBLE);
Datatype STRING = Datatype(Type::STRING);

::testing::AssertionResult compareAST(const AbstractNode &ast1, const AbstractNode &ast2) {
  if (typeid(ast1)!=typeid(ast2)) {
    return ::testing::AssertionFailure() << "AST nodes have different types: " << ast1.toString(false) << " vs "
                                         << ast2.toString(false);
  } else if (typeid(ast1)==typeid(const Assignment &)) {
    // No non-AST attributes
  } else if (typeid(ast1)==typeid(const BinaryExpression &)) {
    auto b1 = dynamic_cast<const BinaryExpression &>(ast1);
    auto b2 = dynamic_cast<const BinaryExpression &>(ast2);
    if (b1.getOperator().toString()!=b2.getOperator().toString()) {
      return ::testing::AssertionFailure() << "BinaryExpressions nodes have different operators: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(Block)) {
    // No non-AST attributes
  } else if (typeid(ast1)==typeid(ExpressionList)) {
    // No non-AST attributes
  } else if (typeid(ast1)==typeid(For)) {
    // No non-AST attributes
  } else if (typeid(ast1)==typeid(const Function &)) {
    auto f1 = dynamic_cast<const Function &>(ast1);
    auto f2 = dynamic_cast<const Function &>(ast2);
    if (f1.getIdentifier()!=f2.getIdentifier()) {
      return ::testing::AssertionFailure() << "Function nodes have different identifiers: " << ast1.toString(false)
                                           << " vs " << ast2.toString(false);
    }
    if (f1.getReturnType()!=f2.getReturnType()) {
      return ::testing::AssertionFailure() << "Function nodes have different return type: " << ast1.toString(false)
                                           << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const FunctionParameter &)) {
    auto f1 = dynamic_cast<const FunctionParameter &>(ast1);
    auto f2 = dynamic_cast<const FunctionParameter &>(ast2);
    if (f1.getIdentifier()!=f2.getIdentifier()) {
      return ::testing::AssertionFailure() << "FunctionParameter nodes have different identifiers: "
                                           << ast1.toString(false)
                                           << " vs " << ast2.toString(false);
    }
    if (f1.getParameterType()!=f2.getParameterType()) {
      return ::testing::AssertionFailure() << "FunctionParameter nodes have different parameter type: "
                                           << ast1.toString(false)
                                           << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const If &)) {
    // No non-AST attributes
  } else if (typeid(ast1)==typeid(IndexAccess)) {
    // No non-AST attributes
  } else if (typeid(ast1)==typeid(const OperatorExpression &)) {
    auto o1 = dynamic_cast<const OperatorExpression &>(ast1);
    auto o2 = dynamic_cast<const OperatorExpression &>(ast2);
    if (o1.getOperator().toString()!=o2.getOperator().toString()) {
      return ::testing::AssertionFailure() << "OperatorExpression nodes have different operators: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(Return)) {
    // No non-AST attributes
  } else if (typeid(ast1)==typeid(const UnaryExpression &)) {
    auto u1 = dynamic_cast<const UnaryExpression &>(ast1);
    auto u2 = dynamic_cast<const UnaryExpression &>(ast2);
    if (u1.getOperator().toString()!=u2.getOperator().toString()) {
      return ::testing::AssertionFailure() << "UnaryExpression nodes have different operators: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const Variable &)) {
    auto v1 = dynamic_cast<const Variable &>(ast1);
    auto v2 = dynamic_cast<const Variable &>(ast2);
    if (v1.getIdentifier()!=v2.getIdentifier()) {
      return ::testing::AssertionFailure() << "Variable nodes have different identifiers: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const VariableDeclaration &)) {
    auto v1 = dynamic_cast<const VariableDeclaration &>(ast1);
    auto v2 = dynamic_cast<const VariableDeclaration &>(ast2);
    if (v1.getDatatype()!=v2.getDatatype()) {
      return ::testing::AssertionFailure() << "VariableDeclaration nodes have different datatypes: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const LiteralBool &)) {
    auto l1 = dynamic_cast<const LiteralBool &>(ast1);
    auto l2 = dynamic_cast<const LiteralBool &>(ast2);
    if (l1.getValue()!=l2.getValue()) {
      return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const LiteralChar &)) {
    auto l1 = dynamic_cast<const LiteralChar &>(ast1);
    auto l2 = dynamic_cast<const LiteralChar &>(ast2);
    if (l1.getValue()!=l2.getValue()) {
      return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const LiteralInt &)) {
    auto l1 = dynamic_cast<const LiteralInt &>(ast1);
    auto l2 = dynamic_cast<const LiteralInt &>(ast2);
    if (l1.getValue()!=l2.getValue()) {
      return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const LiteralFloat &)) {
    auto l1 = dynamic_cast<const LiteralFloat &>(ast1);
    auto l2 = dynamic_cast<const LiteralFloat &>(ast2);
    if (l1.getValue()!=l2.getValue()) {
      return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const LiteralDouble &)) {
    auto l1 = dynamic_cast<const LiteralDouble &>(ast1);
    auto l2 = dynamic_cast<const LiteralDouble &>(ast2);
    if (l1.getValue()!=l2.getValue()) {
      return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else if (typeid(ast1)==typeid(const LiteralString &)) {
    auto l1 = dynamic_cast<const LiteralString &>(ast1);
    auto l2 = dynamic_cast<const LiteralString &>(ast2);
    if (l1.getValue()!=l2.getValue()) {
      return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                           << ast1.toString(false) << " vs " << ast2.toString(false);
    }
  } else {
    throw std::runtime_error("Something bad happened while comparing ASTs.");
  }

  // Compare Children
  if (ast1.countChildren()!=ast2.countChildren())
    return ::testing::AssertionFailure() << "Nodes do not have equal number of children!" << ast1.toString(false) << " has " << ast1.countChildren() << " while " << ast2.toString(false) << " has " << ast2.countChildren();
  auto it1 = ast1.begin();
  auto it2 = ast2.begin();
  for (; it1!=ast1.end() && it2!=ast2.end(); ++it1, ++it2) {
    auto r = compareAST(*it1, *it2);
    if (!r) {
      return ::testing::AssertionFailure() << ast1.toString(true) << " and " << ast2.toString(true)
                                           << " differ in children: " << it1->toString(false) << " vs "
                                           << it2->toString(false)
                                           << "Original issue:" << r.message();
    }
  }
  return ::testing::AssertionSuccess();
}

TEST(ParserTest, emptyString) { /* NOLINT */
  auto b = Parser::parse("");
  // Should be an empty block
  EXPECT_EQ(b->countChildren(), 0);
}

TEST(ParserTest, BinaryExp) { /* NOLINT */
  auto ast = Parser::parse("a = 5 + 6;");

  auto bp = new BinaryExpression(std::make_unique<LiteralInt>(5),
                                 Operator(ArithmeticOp::ADDITION),
                                 std::make_unique<LiteralInt>(6));
  auto assignment = Assignment(std::make_unique<Variable>("a"), std::unique_ptr<BinaryExpression>(bp));
  EXPECT_TRUE(compareAST(*ast->begin(), assignment));
}

TEST(ParserTest, simpleFunction) { /* NOLINT */
  std::string minimal = "public int main() {}";
  std::string params = "public int main(bool b, float f) {}";
  std::string body = "public int main() { return 0; }";

  auto parsed_minimal = Parser::parse(minimal);
  auto parsed_params = Parser::parse(params);
  auto parsed_body = Parser::parse(body);

  auto empty_params = std::vector<std::unique_ptr<FunctionParameter>>();
  auto expected_minimal = Function(INT, "main", std::move(empty_params), std::make_unique<Block>());

  std::vector<std::unique_ptr<FunctionParameter>> parameter_vector;
  parameter_vector.push_back(std::make_unique<FunctionParameter>(BOOL, "b"));
  parameter_vector.push_back(std::make_unique<FunctionParameter>(FLOAT, "f"));
  auto expected_params = Function(INT, "main", std::move(parameter_vector), std::make_unique<Block>());

  empty_params = std::vector<std::unique_ptr<FunctionParameter>>();
  auto body_node = std::make_unique<Block>(std::make_unique<Return>(std::make_unique<LiteralInt>(0)));
  auto expected_body = Function(INT, "main", std::move(empty_params), std::move(body_node));

  // Since parse wraps everything in a Block, get the first element
  EXPECT_TRUE(compareAST(*parsed_minimal->begin(), expected_minimal));
  EXPECT_TRUE(compareAST(*parsed_params->begin(), expected_params));
  EXPECT_TRUE(compareAST(*parsed_body->begin(), expected_body));
}

TEST(ParserTest, IfStatementThenOnly) { /* NOLINT */
  const char *programCode = R""""(
    public int main(int a) {
      if (a > 5) {
        return 1;
      }
      return 0;
    }
    )"""";
  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  auto ifStatement = std::make_unique<If>
      (std::move(std::make_unique<BinaryExpression>(
          std::move(std::make_unique<Variable>("a")),
          Operator(GREATER),
          std::move(std::make_unique<LiteralInt>(5)))),
       std::move(std::make_unique<Block>(
           std::move(std::make_unique<Return>
                         (std::move(std::make_unique<LiteralInt>(1)))))));
  auto returnStatement = std::make_unique<Return>(std::move(std::make_unique<LiteralInt>(0)));

  std::vector<std::unique_ptr<AbstractStatement>> blockStmts;
  blockStmts.emplace_back(std::move(ifStatement));
  blockStmts.emplace_back(std::move(returnStatement));
  auto expected_body = std::make_unique<Block>(std::move(blockStmts));

  std::vector<std::unique_ptr<FunctionParameter>> fParams;
  fParams.emplace_back(std::move(std::make_unique<FunctionParameter>(Datatype(Type::INT, false), "a")));
  auto funcParams = std::vector<std::unique_ptr<FunctionParameter>>(std::move(fParams));
  auto expected = new Function(Datatype(Type::INT, false), "main", std::move(funcParams), std::move(expected_body));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, IfStatementThenOnly_WithoutBlock) { /* NOLINT */
  const char *programCode = R""""(
    public int main(int a) {
      if (a > 5) return 1;
      return 0;
    }
    )"""";
  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  auto ifStatement = std::make_unique<If>
      (std::move(std::make_unique<BinaryExpression>(
          std::move(std::make_unique<Variable>("a")),
          Operator(GREATER),
          std::move(std::make_unique<LiteralInt>(5)))),
       std::move(std::make_unique<Block>(
           std::move(std::make_unique<Return>
                         (std::move(std::make_unique<LiteralInt>(1)))))));
  auto returnStatement = std::make_unique<Return>(std::move(std::make_unique<LiteralInt>(0)));

  std::vector<std::unique_ptr<AbstractStatement>> blockStmts;
  blockStmts.emplace_back(std::move(ifStatement));
  blockStmts.emplace_back(std::move(returnStatement));
  auto expected_body = std::make_unique<Block>(std::move(blockStmts));

  std::vector<std::unique_ptr<FunctionParameter>> fParams;
  fParams.emplace_back(std::move(std::make_unique<FunctionParameter>(Datatype(Type::INT, false), "a")));
  auto funcParams = std::vector<std::unique_ptr<FunctionParameter>>(std::move(fParams));
  auto expected = new Function(Datatype(Type::INT, false), "main", std::move(funcParams), std::move(expected_body));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, IfElseStatement) { /* NOLINT */
  const char *programCode = R""""(
    public int main(int a) {
      if (a > 5) {
        return 111;
      } else {
        return 0;
      }
    }
    )"""";
  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  auto ifStatement = std::make_unique<If>
      (std::move(std::make_unique<BinaryExpression>(
          std::move(std::make_unique<Variable>("a")),
          Operator(GREATER),
          std::move(std::make_unique<LiteralInt>(5)))),
       std::move(std::make_unique<Block>(
           std::move(std::make_unique<Return>
                         (std::move(std::make_unique<LiteralInt>(111)))))),
       std::move(std::make_unique<Block>(
           std::move(std::make_unique<Return>
                         (std::move(std::make_unique<LiteralInt>(0)))))));

  std::vector<std::unique_ptr<AbstractStatement>> blockStmts;
  blockStmts.emplace_back(std::move(ifStatement));
  auto expected_body = std::make_unique<Block>(std::move(blockStmts));

  std::vector<std::unique_ptr<FunctionParameter>> fParams;
  fParams.emplace_back(std::move(std::make_unique<FunctionParameter>(Datatype(Type::INT, false), "a")));
  auto funcParams = std::vector<std::unique_ptr<FunctionParameter>>(std::move(fParams));
  auto expected = new Function(Datatype(Type::INT, false), "main", std::move(funcParams), std::move(expected_body));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

