#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/Literal.h"
#include "gtest/gtest.h"

TEST(FunctionParameterTest, values_ValuesGivenInCtorAreRetrievable) {
  // FunctionParameter statements are created with an identifier and type
  // TODO: This test simply confirms that they are retrievable later
}

TEST(FunctionParameterTest, SetAndGet) {
  // TODO: This test simply checks that identifier and type can be set and get correctly.
}

TEST(FunctionParameterTest, CopyCtorCopiesValue) {
  // TODO: When copying a FunctionParameter, the new object should contain a (deep) copy of the identifier and type
}

TEST(FunctionParameterTest, CopyAssignmentCopiesValue) {
  // TODO: When copying a FunctionParameter, the new object should contain a copy of the identifier and type
}

TEST(FunctionParameterTest, MoveCtorPreservesValue) {
  // TODO: When moving a FunctionParameter, the new object should contain the same identifier and type
}

TEST(FunctionParameterTest, MoveAssignmentPreservesValue) {
  // TODO: When moving a FunctionParameter, the new object should contain the same identifier and type
}

TEST(FunctionParameterTest, countChildrenReportsCorrectNumber) {
  // TODO: This tests checks that countChildren delivers the correct number
}

TEST(FunctionParameterTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

}

TEST(FunctionParameterTest, JsonOutputTest) { /* NOLINT */
 // TODO: Verify JSON output
}