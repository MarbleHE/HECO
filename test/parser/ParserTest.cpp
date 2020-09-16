#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/parser/Parser.h"
#include "gtest/gtest.h"

Datatype BOOL = Datatype(Type::BOOL);
Datatype CHAR = Datatype(Type::CHAR);
Datatype INT = Datatype(Type::INT);
Datatype FLOAT = Datatype(Type::FLOAT);
Datatype DOUBLE = Datatype(Type::DOUBLE);
Datatype STRING = Datatype(Type::STRING);

bool compareAST(const AbstractNode& ast1, const AbstractNode& ast2) {
  //TODO: Write helper function to compare two AST's structure & content without caring about IDs
  return false;
}



TEST(ParserTest, emptyString) {
  auto b = Parser::parse("");
  // Should be an empty block
  EXPECT_EQ(b->countChildren(),0);
}

TEST(ParserTest, BinaryExp) {
  auto ast = Parser::parse("a = 5 + 6");

  auto bp = new BinaryExpression(std::make_unique<LiteralInt>(5), Operator(ArithmeticOp::ADDITION), std::make_unique<LiteralInt>(6));
  auto assignment = Assignment(std::make_unique<Variable>("a"), std::unique_ptr<BinaryExpression>(bp));
  EXPECT_TRUE(compareAST(*ast, assignment));
}

TEST(ParserTest, simpleFunction) {
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
  auto body_node = std::make_unique<Block>(std::make_unique<Return>(std::make_unique<LiteralBool>(0)));
  auto expected_body = Function(INT, "main", std::move(empty_params), std::move(body_node));


  EXPECT_TRUE( compareAST(*parsed_minimal, expected_minimal));
  EXPECT_TRUE( compareAST(*parsed_params, expected_params));
  EXPECT_TRUE( compareAST(*parsed_body, expected_body));
}
