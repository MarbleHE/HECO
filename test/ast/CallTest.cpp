#include "abc/ast/Call.h"
#include "gtest/gtest.h"

TEST(CallTest, values_ValuesGivenInCtorAreRetrievable) {
  // Call statements are created with an identifier and a list of Expressions for the arguments (but not an ExpressionList)
  // TODO: This test simply confirms that they are retrievable later
}

TEST(CallTest, SetAndGet) {
  // TODO: This test simply checks that identifier and list of Expressions can be set and get correctly.
}

TEST(CallTest, CopyCtorCopiesValue) {
  // TODO: When copying a Call, the new object should contain a (deep) copy of the identifier and list of Expressions
}

TEST(CallTest, CopyAssignmentCopiesValue) {
  // TODO: When copying a Call, the new object should contain a copy of the identifier and list of Expressions
}

TEST(CallTest, MoveCtorPreservesValue) {
  // TODO: When moving a Call, the new object should contain the same identifier and list of Expressions
}

TEST(CallTest, MoveAssignmentPreservesValue) {
  // TODO: When moving a Call, the new object should contain the same identifier and list of Expressions
}

TEST(CallTest, countChildrenReportsCorrectNumber) {
  // TODO: This tests checks that countChildren delivers the correct number
}

TEST(CallTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

}

TEST(CallTest, JsonOutputTest) { /* NOLINT */
 // TODO: Verify JSON output
}