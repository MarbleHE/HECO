#include "heco/legacy_ast/ast/UnaryExpression.h"
#include "gtest/gtest.h"
#include "heco/legacy_ast/ast/Literal.h"

TEST(UnaryExpressionTest, values_ValuesGivenInCtorAreRetrievable)
{
    // Unary expressions are created with an operand and an operator
    // TODO: This test simply confirms that they are retrievable later
}

TEST(UnaryExpressionTest, SetAndGet)
{
    // TODO: This test simply checks that operands and operator can be set and get correctly.
}

TEST(UnaryExpressionTest, CopyCtorCopiesValue)
{
    // TODO: When copying a UnaryExpression, the new object should contain a (deep) copy of the operands and operator
}

TEST(UnaryExpressionTest, CopyAssignmentCopiesValue)
{
    // TODO: When copying a UnaryExpression, the new object should contain a copy of the operands and operator
}

TEST(UnaryExpressionTest, MoveCtorPreservesValue)
{
    // TODO: When moving a UnaryExpression, the new object should contain the same operands and operator
}

TEST(UnaryExpressionTest, MoveAssignmentPreservesValue)
{
    // TODO: When moving a UnaryExpression, the new object should contain the same operands and operator
}

TEST(UnaryExpressionTest, countChildrenReportsCorrectNumber)
{
    // TODO: This tests checks that countChildren delivers the correct number
}

TEST(UnaryExpressionTest, node_iterate_children)
{
    // TODO: This test checks that we can iterate correctly through the children
    // Even if some of the elements are null (in which case they should not appear)
}

TEST(UnaryExpressionTest, JsonOutputTest)
{ /* NOLINT */
    // TODO: Verify JSON output
}