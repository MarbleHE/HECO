#include <include/ast_opt/parser/Parser.h>
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/Literal.h"
#include "gtest/gtest.h"

TEST(FunctionTest, values_ValuesGivenInCtorAreRetrievable) {
  // Functions are created with a return type, identifier, parameter vector and body
  // TODO: This test simply confirms that they are retrievable later
}

TEST(FunctionTest, SetAndGet) {
  // TODO: This test simply checks that return type, identifier, parameter vector and body can be set and get correctly.
}

TEST(FunctionTest, CopyCtorCopiesValue) {
  // TODO: When copying a Function, the new object should contain a (deep) copy of the return type, identifier, parameter vector and body
}

TEST(FunctionTest, CopyAssignmentCopiesValue) {
  // TODO: When copying a Function, the new object should contain a copy of the return type, identifier, parameter vector and body
}

TEST(FunctionTest, MoveCtorPreservesValue) {
  // TODO: When moving a Function, the new object should contain the same return type, identifier, parameter vector and body
}

TEST(FunctionTest, MoveAssignmentPreservesValue) {
  // TODO: When moving a Function, the new object should contain the same return type, identifier, parameter vector and body
}

TEST(FunctionTest, countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
  FunctionParameter functionParameter(Datatype(Type::BOOL), "foo");
  std::vector<std::unique_ptr<FunctionParameter>> paramVec;
  paramVec.push_back(std::make_unique<FunctionParameter>(functionParameter));
  std::unique_ptr<Block> block = std::make_unique<Block>(variableDeclaration1.clone());
  Function f(Datatype(Type::BOOL), "main", std::move(paramVec), std::move(block));
  auto reported_count = f.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for (auto &c: f) {
    c = c; // Use c to suppress warning
    ++actual_count;
  }

  EXPECT_EQ(reported_count, actual_count);
}

TEST(FunctionTest, node_iterate_children_multipleParameters) {
  // This test checks that we can iterate correctly through the children
  const char *inputChars = R""""(
    public int main(int a, int z, int v) {
      return a;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  auto function = inputAST->begin();
  auto it = function->begin();

  EXPECT_NE(dynamic_cast<FunctionParameter *>(&(*it)), nullptr);
  it++;
  EXPECT_NE(dynamic_cast<FunctionParameter *>(&(*it)), nullptr);
  it++;
  EXPECT_NE(dynamic_cast<FunctionParameter *>(&(*it)), nullptr);
  it++;
  EXPECT_NE(dynamic_cast<Block *>(&(*it)), nullptr);
  it++;
  EXPECT_EQ(it, function->end());
}

TEST(FunctionTest, node_iterate_children_singleParameter) {
  // This test checks that we can iterate correctly through the children
  const char *inputChars = R""""(
    public int main(int a) {
      return a;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  auto function = inputAST->begin();
  auto it = function->begin();

  EXPECT_NE(dynamic_cast<FunctionParameter *>(&(*it)), nullptr);
  it++;
  EXPECT_NE(dynamic_cast<Block *>(&(*it)), nullptr);
  it++;
  EXPECT_EQ(it, function->end());
}

TEST(FunctionTest, node_iterate_children_noParameter) {
  // This test checks that we can iterate correctly through the children
  const char *inputChars = R""""(
    public int main() {
      return a;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  auto function = inputAST->begin();
  auto it = function->begin();

  EXPECT_NE(dynamic_cast<Block *>(&(*it)), nullptr);
  it++;
  EXPECT_EQ(it, function->end());
}

TEST(FunctionTest, JsonOutputTest) { /* NOLINT */
  // TODO: Verify JSON output
}
