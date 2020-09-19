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
  Function f(Datatype(Type::BOOL), "main",std::move(paramVec), std::move(block));
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

TEST(FunctionTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children

  Function f(Datatype(Type::BOOL), "main",{}, std::make_unique<Block>());
  for(auto& c:f) {
    // Should have only one Block in here
    EXPECT_EQ(c.toString(true),Block().toString(true));
  }
  // Even if some of the elements are null (in which case they should not appear)



}

TEST(FunctionTest, JsonOutputTest) { /* NOLINT */
 // TODO: Verify JSON output
}