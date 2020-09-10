#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Literal.h"
#include "gtest/gtest.h"

TEST(ForTest, values_ValuesGivenInCtorAreRetrievable) {
  // For statements are created with an initializer, condition, update and body
  // TODO: This test simply confirms that they are retrievable later
}

TEST(ForTest, SetAndGet) {
  // TODO: This test simply checks that initializer, condition, update and body can be set and get correctly.
}

TEST(ForTest, CopyCtorCopiesValue) {
  // TODO: When copying a For, the new object should contain a (deep) copy of the initializer, condition, update and body
}

TEST(ForTest, CopyAssignmentCopiesValue) {
  // TODO: When copying a For, the new object should contain a copy of the initializer, condition, update and body
}

TEST(ForTest, MoveCtorPreservesValue) {
  // TODO: When moving a For, the new object should contain the same initializer, condition, update and body
}

TEST(ForTest, MoveAssignmentPreservesValue) {
  // TODO: When moving a For, the new object should contain the same initializer, condition, update and body
}

TEST(ForTest, countChildrenReportsCorrectNumber) {
  // TODO: This tests checks that countChildren delivers the correct number
}

TEST(ForTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

}

TEST(ForTest, JsonOutputTest) { /* NOLINT */
 // TODO: Verify JSON output
}