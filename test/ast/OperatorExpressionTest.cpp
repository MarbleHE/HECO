#include "abc/ast/OperatorExpression.h"
#include "abc/ast/Literal.h"
#include "abc/ast/Variable.h"
#include "abc/ast/VariableDeclaration.h"
#include "gtest/gtest.h"

TEST(OperatorExpressionTest, values_ValuesGivenInCtorAreRetrievable) {
  // TODO: This test simply confirms that operands supplied via Ctor are retrievable later
}

TEST(OperatorExpressionTest, CopyCtorCopiesValue) {
  // TODO:  When copying a OperatorExpression, the new object should contain a (deep) copy of all the operands
}

TEST(OperatorExpressionTest, CopyAssignmentCopiesValue) {
  // TODO: When copying a OperatorExpression, the new object should contain a (deep) copy of all the operands
}

TEST(OperatorExpressionTest, MoveCtorPreservesValue) {
  // TODO: When moving a OperatorExpression, the new object should contain the same operands
}

TEST(OperatorExpressionTest, MovedAssignmentPreservesValue) {
  // TODO: When moving a OperatorExpression, the new object should contain the same operands
}

TEST(OperatorExpressionTest, NullStatementRemoval) {
  // TODO: Removing null operands should not affect the other children
}

TEST(OperatorExpressionTest, CountChildrenReportsCorrectNumber) {
  // TODO: This tests checks that countChildren delivers the correct number
}

TEST(OperatorExpressionTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear
}

TEST(OperatorExpressionTest, appendStatement) {
  // TODO: This test checks that we can append a Statement to the OperatorExpression
}

TEST(OperatorExpressionTest, prependStatement) {
  //  TODO: This test checks that we can prepend an Operand to the OperatorExpression
}

TEST(OperatorExpressionTest, JsonOutputTest) { /* NOLINT */
  // TODO: Json output test
}