#include "abc/ast/ExpressionList.h"
#include "gtest/gtest.h"

TEST(ExpressionListTest, values_ValuesGivenInCtorAreRetrievable) {
  // TODO: This test simply confirms that operands supplied via Ctor are retrievable later
}

TEST(ExpressionListTest, CopyCtorCopiesValue) {
  // TODO:  When copying a ExpressionList, the new object should contain a (deep) copy of all the operands
}

TEST(ExpressionListTest, CopyAssignmentCopiesValue) {
  // TODO: When copying a ExpressionList, the new object should contain a (deep) copy of all the operands
}

TEST(ExpressionListTest, MoveCtorPreservesValue) {
  // TODO: When moving a ExpressionList, the new object should contain the same operands
}

TEST(ExpressionListTest, MovedAssignmentPreservesValue) {
  // TODO: When moving a ExpressionList, the new object should contain the same operands
}

TEST(ExpressionListTest, NullStatementRemoval) {
  // TODO: Removing null operands should not affect the other children
}

TEST(ExpressionListTest, CountChildrenReportsCorrectNumber) {
  // TODO: This tests checks that countChildren delivers the correct number
}

TEST(ExpressionListTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear
}

TEST(ExpressionListTest, appendStatement) {
  // TODO: This test checks that we can append a Statement to the ExpressionList
}

TEST(ExpressionListTest, prependStatement) {
  //  TODO: This test checks that we can prepend an Operand to the ExpressionList
}

TEST(ExpressionListTest, JsonOutputTest) { /* NOLINT */
  // TODO: Json output test
}

TEST(ExpressionListTest, JsonInputTest) { /* NOLINT */
  // TODO: Json input test
}