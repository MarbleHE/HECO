#include "ast_opt/ast/IndexAccess.h"
#include "gtest/gtest.h"

TEST(IndexAccessTest, values_ValuesGivenInCtorAreRetrievable) {
  // Unary expressions are created with a target and index
  // TODO: This test simply confirms that they are retrievable later
}

TEST(IndexAccessTest, SetAndGet) {
  // TODO: This test simply checks that target and index can be set and get correctly.
}

TEST(IndexAccessTest, CopyCtorCopiesValue) {
  // TODO: When copying a IndexAccess, the new object should contain a (deep) copy of the target and index
}

TEST(IndexAccessTest, CopyAssignmentCopiesValue) {
  // TODO: When copying a IndexAccess, the new object should contain a copy of the target and index
}

TEST(IndexAccessTest, MoveCtorPreservesValue) {
  // TODO: When moving a IndexAccess, the new object should contain the same target and index
}

TEST(IndexAccessTest, MoveAssignmentPreservesValue) {
  // TODO: When moving a IndexAccess, the new object should contain the same target and index
}

TEST(IndexAccessTest, countChildrenReportsCorrectNumber) {
  // TODO: This tests checks that countChildren delivers the correct number
}

TEST(IndexAccessTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

}

TEST(IndexAccessTest, JsonOutputTest) { /* NOLINT */
 // TODO: Verify JSON output
}

TEST(IndexAccessTest, JsonInputTest) { /* NOLINT */
// TODO: Verify JSON input
}