#include "heco/legacy_ast/ast/BinaryExpression.h"
#include <heco/legacy_ast/ast_parser/Parser.h>
#include "ASTComparison.h"
#include "gtest/gtest.h"
#include "heco/legacy_ast/ast/Literal.h"

TEST(BinaryExpressionTest, values_ValuesGivenInCtorAreRetrievable)
{
    // Binary expressions are created with two operands and an operator
    // TODO: This test simply confirms that they are retrievable later
}

TEST(BinaryExpressionTest, SetAndGet)
{
    // TODO: This test simply checks that operands and operator can be set and get correctly.
}

TEST(BinaryExpressionTest, CopyCtorCopiesValue)
{
    // TODO: When copying a BinaryExpression, the new object should contain a (deep) copy of the operands and operator
}

TEST(BinaryExpressionTest, CopyAssignmentCopiesValue)
{
    // TODO: When copying a BinaryExpression, the new object should contain a copy of the operands and operator
}

TEST(BinaryExpressionTest, MoveCtorPreservesValue)
{
    // TODO: When moving a BinaryExpression, the new object should contain the same operands and operator
}

TEST(BinaryExpressionTest, MoveAssignmentPreservesValue)
{
    // TODO: When moving a BinaryExpression, the new object should contain the same operands and operator
}

TEST(BinaryExpressionTest, countChildrenReportsCorrectNumber)
{
    // TODO: This tests checks that countChildren delivers the correct number
}

TEST(BinaryExpressionTest, node_iterate_children)
{
    // TODO: This test checks that we can iterate correctly through the children
    // Even if some of the elements are null (in which case they should not appear)
}

TEST(BinaryExpressionTest, JsonOutputTest)
{ /* NOLINT */
    // TODO: Verify JSON output
}

TEST(BinaryExpressionTest, JsonInputTest)
{ /* NOLINT */
    BinaryExpression expected(
        std::make_unique<LiteralInt>(1), Operator(ArithmeticOp::ADDITION), std::make_unique<LiteralInt>(1));

    std::string binExprJson = R""""({
    "left": {
      "type": "LiteralInt",
      "value": 1
    },
    "operator": "+",
    "right": {
      "type": "LiteralInt",
      "value": 1
    },
    "type": "BinaryExpression"
  })"""";

    auto parsed = Parser::parseJson(binExprJson);

    ASSERT_TRUE(compareAST(*parsed, expected));
}