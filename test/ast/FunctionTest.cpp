#include "ast_opt/ast/Function.h"
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
  // TODO: This tests checks that countChildren delivers the correct number
}

TEST(FunctionTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

}

TEST(FunctionTest, JsonOutputTest) { /* NOLINT */
 // TODO: Verify JSON output
}