#include "ast_opt/ast/Function.h"
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

auto empty_params = std::vector<std::unique_ptr<FunctionParameter>>();

TEST(ParserTest, function) {
  //TODO: Write working parser tests
//  std::string minimal = "int main() {}";
//  std::string params = "int main(bool b, float f) {}";
//  std::string body = "int main() { return 0; }";
//
//  auto parsed_minimal = Parser::parse(minimal);
//  auto parsed_params = Parser::parse(params);
//  auto parsed_body = Parser::parse(body);
//
//  auto expected_minimal = Function(INT, "main", empty_params, std::make_unique<Block>());
//
//  auto parameter_vector =
//      {std::make_unique<FunctionParameter>("b", BOOL), std::make_unique<FunctionParameter>("f", FLOAT)};
//  auto expected_params = Function(INT, "main", parameter_vector, std::make_unique<Block>());
//
//  auto body_node = std::make_unique<Block>(std::make_unique<Return>(std::make_unique<LiteralBool>(0)));
//  auto expected_body = Function(INT, "main", empty_params, std::move(body_node));

  //TODO: Write helper function to compare two AST's structure & content without caring about IDs
  //EXPECT_EQ(*parsed_minimal, expected_minimal);
  //EXPECT_EQ(*parsed_params, expected_params);
  //EXPECT_EQ(*parsed_body, expected_body);
}

TEST(ParserTest, recognizeInputTest) {
  std::string path = __FILE__;
  path = path.substr(0, path.find_last_of("/\\") + 1) + "test.stk";

//  using namespace stork;
//
//  module m;
//
//  add_standard_functions(m);
//
//  auto s_main = m.create_public_function_caller<void>("main");
//
//  if (m.try_load(path.c_str(), &std::cerr)) {
//    s_main();
//  }

}
