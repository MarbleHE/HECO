#include <typeinfo>
#include <include/ast_opt/parser/Errors.h>

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
#include "../ASTComparison.h"

Datatype BOOL = Datatype(Type::BOOL);
Datatype CHAR = Datatype(Type::CHAR);
Datatype INT = Datatype(Type::INT);
Datatype FLOAT = Datatype(Type::FLOAT);
Datatype DOUBLE = Datatype(Type::DOUBLE);
Datatype STRING = Datatype(Type::STRING);

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

TEST(ParserTest, IfElseIfStatements) { /* NOLINT */
  // The generated AST is supposed to look as follow:
  // public int main(int a) {
  //   if (a < 0) {
  //     return -1;
  //   } else {
  //     if (a == 0) {
  //       return 1000;
  //     } else {
  //       if (a > 4256) {
  //         return 3434;
  //       }
  //     }
  //   }
  //   return 0;
  // }
  const char *programCode = R""""(
    public int main(int a) {
      if (a < 0) {
        return -1;
      } else if (a == 0) {
        return 1000;
      } else if (a > 4256) {
        return 3434;
      }
      return 0;
    }
    )"""";

  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  auto ifStatement4256 = std::make_unique<If>
      (std::move(std::make_unique<BinaryExpression>(
          std::move(std::make_unique<Variable>("a")),
          Operator(GREATER),
          std::move(std::make_unique<LiteralInt>(4256)))),
       std::move(std::make_unique<Block>(
           std::move(std::make_unique<Return>
                         (std::move(std::make_unique<LiteralInt>(3434)))))));

  auto ifStatementEqual0 = std::make_unique<If>
      (std::move(std::make_unique<BinaryExpression>(
          std::move(std::make_unique<Variable>("a")),
          Operator(EQUAL),
          std::move(std::make_unique<LiteralInt>(0)))),
       std::move(std::make_unique<Block>(
           std::move(std::make_unique<Return>
                         (std::move(std::make_unique<LiteralInt>(1000)))))),
       std::move(std::make_unique<Block>(std::move(ifStatement4256))));

  auto ifStatementLess0 = std::make_unique<If>
      (std::move(std::make_unique<BinaryExpression>(
          std::move(std::make_unique<Variable>("a")),
          Operator(LESS),
          std::move(std::make_unique<LiteralInt>(0)))),
       std::move(std::make_unique<Block>(
           std::move(std::make_unique<Return>
                         (std::move(std::make_unique<LiteralInt>(-1)))))),
       std::move(std::make_unique<Block>(std::move(ifStatementEqual0))));

  auto returnStatement = std::make_unique<Return>(std::move(std::make_unique<LiteralInt>(0)));

  std::vector<std::unique_ptr<AbstractStatement>> blockStmts;
  blockStmts.emplace_back(std::move(ifStatementLess0));
  blockStmts.emplace_back(std::move(returnStatement));
  auto expected_body = std::make_unique<Block>(std::move(blockStmts));

  std::vector<std::unique_ptr<FunctionParameter>> fParams;
  fParams.emplace_back(std::move(std::make_unique<FunctionParameter>(Datatype(Type::INT, false), "a")));
  auto funcParams = std::vector<std::unique_ptr<FunctionParameter>>(std::move(fParams));
  auto expected = new Function(Datatype(Type::INT, false), "main", std::move(funcParams), std::move(expected_body));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, ForStatement) { /* NOLINT */
  const char *programCode = R""""(
    public secret int computeSum(int bound) {
      int sum = 0;
      for (int i = 0; i < bound; i = i + 1) {
        sum = sum + i;
      }
      return sum;
    }
    )"""";

  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  // int sum = 0;
  auto declarationSum =
      std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                            std::make_unique<Variable>("sum"),
                                            std::make_unique<LiteralInt>(0));

  // for (int i = 0; i < bound; i = i + 1) { sum = sum + i; }
  auto forStatement = std::make_unique<For>(
      // int i = 0
      std::make_unique<Block>(std::make_unique<VariableDeclaration>(
          Datatype(Type::INT, false), std::make_unique<Variable>("i"), std::make_unique<LiteralInt>(0))),
      // i < bound
      std::make_unique<BinaryExpression>(
          std::make_unique<Variable>("i"), Operator(LESS), std::make_unique<Variable>("bound")),
      // i = i + 1
      std::make_unique<Block>(std::make_unique<Assignment>(
          std::make_unique<Variable>("i"), std::make_unique<BinaryExpression>(
              std::make_unique<Variable>("i"), Operator(ADDITION), std::make_unique<LiteralInt>(1)))),
      // { sum = sum + i; }
      std::make_unique<Block>(std::make_unique<Assignment>(
          std::make_unique<Variable>("sum"),
          std::make_unique<BinaryExpression>(
              std::make_unique<Variable>("sum"), Operator(ADDITION), std::make_unique<Variable>("i")))));

  // return sum
  auto returnStmt = std::make_unique<Return>(std::make_unique<Variable>("sum"));

  std::vector<std::unique_ptr<AbstractStatement>> functionBlockStatements;
  functionBlockStatements.emplace_back(std::move(declarationSum));
  functionBlockStatements.emplace_back(std::move(forStatement));
  functionBlockStatements.emplace_back(std::move(returnStmt));

  std::vector<std::unique_ptr<FunctionParameter>> functionParameters;
  functionParameters.emplace_back(
      std::make_unique<FunctionParameter>(Datatype(Type::INT, false), "bound"));

  auto expected = new Function(Datatype(Type::INT, true),
                               "computeSum",
                               std::move(functionParameters),
                               std::move(std::make_unique<Block>(std::move(functionBlockStatements))));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, IgnoreComments) { /* NOLINT */
  const char *programCode = R""""(
      // declare and initialize a variable
      int i = 0;  /* variable's value: 0 */
    )"""";

  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  auto expected = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                        std::make_unique<Variable>("i"),
                                                        std::make_unique<LiteralInt>(0));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, MatrixDeclaration_simple) { /* NOLINT */
  const char *programCode = R""""(
    public void main() {
      int scalar = 2;
      int vec = {3, 4, 9, 2, 1};
    }
    )"""";

  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

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

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, MatrixDeclaration_multiDimensional) { /* NOLINT */
  const char *programCode = R""""(
    public void main() {
      int vec = { {3, 4}, {9, 2}, {1} };
    }
    )"""";

  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  // int vec = { {3, 4}, {9, 2}, {1} };
  typedef std::vector<std::unique_ptr<AbstractExpression>> listOfExpressions;

  listOfExpressions exprs1;
  exprs1.emplace_back(std::make_unique<LiteralInt>(3));
  exprs1.emplace_back(std::make_unique<LiteralInt>(4));
  auto exprList1 = std::make_unique<ExpressionList>(std::move(exprs1));

  listOfExpressions exprs2;
  exprs2.emplace_back(std::make_unique<LiteralInt>(9));
  exprs2.emplace_back(std::make_unique<LiteralInt>(2));
  auto exprList2 = std::make_unique<ExpressionList>(std::move(exprs2));

  listOfExpressions exprs3;
  exprs3.emplace_back(std::make_unique<LiteralInt>(1));
  auto exprList3 = std::make_unique<ExpressionList>(std::move(exprs3));

  listOfExpressions exprs;
  exprs.emplace_back(std::move(exprList1));
  exprs.emplace_back(std::move(exprList2));
  exprs.emplace_back(std::move(exprList3));
  auto exprsList = std::make_unique<ExpressionList>(std::move(exprs));

  auto declarationVec = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                              std::make_unique<Variable>("vec"),
                                                              std::move(exprsList));

  // public void main() { ... }
  auto expected = std::make_unique<Function>(Datatype(Type::VOID),
                                             "main",
                                             std::move(std::vector<std::unique_ptr<FunctionParameter>>()),
                                             std::move(std::make_unique<Block>(std::move(declarationVec))));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, MatrixAssignment) { /* NOLINT */
  const char *programCode = R""""(
    public void main() {
      int vec = {3, 4, 9, 2, 1};
      vec[3] = 0;
    }
    )"""";

  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

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

  auto assignVec = std::make_unique<Assignment>(std::make_unique<IndexAccess>(std::make_unique<Variable>("vec"),
                                                                              std::make_unique<LiteralInt>(3)),
                                                std::make_unique<LiteralInt>(0));

  // public void main() { ... }
  std::vector<std::unique_ptr<AbstractStatement>> statements;
  statements.push_back(std::move(declarationVec));
  statements.push_back(std::move(assignVec));
  auto statementBlock = std::make_unique<Block>(std::move(statements));
  auto expected = std::make_unique<Function>(Datatype(Type::VOID),
                                             "main",
                                             std::move(std::vector<std::unique_ptr<FunctionParameter>>()),
                                             std::move(statementBlock));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, MatrixAssignment_invalid) { /* NOLINT */
  const char *programCode = R""""(
      int sum[5] = {3, 4, 9, 2, 1};
      return sum;
    )"""";
  auto code = std::string(programCode);
  ASSERT_THROW(Parser::parse(code), stork::Error::exception);
}

TEST(ParserTest, MatrixDeclaration_brackets) { /* NOLINT */
  const char *programCode = R""""(
    public void main() {
      int scalar[] = 2;
    }
    )"""";

  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  // int scalar[] = 2;
  auto declarationScalar = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                                 std::move(std::make_unique<Variable>("scalar")),
                                                                 std::move(std::make_unique<LiteralInt>(2)));

  // public void main() { ... }
  std::vector<std::unique_ptr<AbstractStatement>> statements;
  statements.push_back(std::move(declarationScalar));
  auto statementBlock = std::make_unique<Block>(std::move(statements));
  auto expected = std::make_unique<Function>(Datatype(Type::VOID),
                                             "main",
                                             std::move(std::vector<std::unique_ptr<FunctionParameter>>()),
                                             std::move(statementBlock));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, MatrixDeclaration_fixArraySizeNotSupported) { /* NOLINT */
  const char *programCode = R""""(
    public void main() {
      int scalar[0] = 2;
    }
    )"""";

  auto code = std::string(programCode);
  EXPECT_THROW(Parser::parse(code), stork::Error::exception);
}

TEST(ParserTest, fhe_expression) { /* NOLINT */
  const char *programCode = R""""(
    __input2__ = __input2__ +++ __input3__;
    )"""";

  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  auto expr = std::make_unique<BinaryExpression>(std::make_unique<Variable>("__input2__"),
                                                 Operator(ArithmeticOp::FHE_ADDITION),
                                                 std::make_unique<Variable>("__input3__"));
  auto assignment = Assignment(std::make_unique<Variable>("__input2__"), std::move(expr));

  EXPECT_TRUE(compareAST(*parsed->begin(), assignment));
}

TEST(ParserTest, parenthesisExpression) { /* NOLINT */
  const char *programCode = R""""(
      public int main(int b) {
        int a = (5+7)*(b<10);
        return a;
      }
    )"""";
  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  // Function block
  std::vector<std::unique_ptr<AbstractStatement>> blockStatements;

  auto expr = std::make_unique<BinaryExpression>(
      std::make_unique<BinaryExpression>(std::make_unique<LiteralInt>(5),
                                         Operator(ADDITION),
                                         std::make_unique<LiteralInt>(7)),
      Operator(MULTIPLICATION),
      std::make_unique<BinaryExpression>(std::make_unique<Variable>("b"),
                                         Operator(LESS),
                                         std::make_unique<LiteralInt>(10))
  );

  blockStatements.push_back(
      std::make_unique<VariableDeclaration>(Datatype(Type::INT),
                                            std::make_unique<Variable>("a"),
                                            std::move(expr)));

  blockStatements.push_back(std::make_unique<Return>(std::make_unique<Variable>("a")));

  auto block = std::make_unique<Block>(std::move(blockStatements));

  // Function
  std::vector<std::unique_ptr<FunctionParameter>> fparams;
  fparams.push_back(std::make_unique<FunctionParameter>(Datatype(Type::INT), "b"));
  auto expected = std::make_unique<Function>(Datatype(Type::INT), "main", std::move(fparams), std::move(block));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}

TEST(ParserTest, secretKeyword) { /* NOLINT */
  const char *programCode = R""""(
    public secret int main(secret int a) {
      secret int b = 11;
      return a+b;
    }
    )"""";
  auto code = std::string(programCode);
  auto parsed = Parser::parse(code);

  auto varDecl = std::make_unique<VariableDeclaration>(Datatype(Type::INT, true),
                                                       std::make_unique<Variable>("b"),
                                                       std::make_unique<LiteralInt>(11));
  auto returnStatement = std::make_unique<Return>(std::make_unique<BinaryExpression>(
      std::make_unique<Variable>("a"), Operator(ADDITION), std::make_unique<Variable>("b")
  ));
  std::vector<std::unique_ptr<AbstractStatement>> blockStmts;
  blockStmts.emplace_back(std::move(varDecl));
  blockStmts.emplace_back(std::move(returnStatement));
  auto expected_body = std::make_unique<Block>(std::move(blockStmts));

  std::vector<std::unique_ptr<FunctionParameter>> fParams;
  fParams.emplace_back(std::move(std::make_unique<FunctionParameter>(Datatype(Type::INT, true), "a")));
  auto funcParams = std::vector<std::unique_ptr<FunctionParameter>>(std::move(fParams));
  auto expected = new Function(Datatype(Type::INT, false), "main", std::move(funcParams), std::move(expected_body));

  EXPECT_TRUE(compareAST(*parsed->begin(), *expected));
}
