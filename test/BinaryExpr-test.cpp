#include <include/ast/LiteralInt.h>
#include <BinaryExpr.h>
#include "gtest/gtest.h"

TEST(BinaryExprTests, leftOperand) {
    LiteralInt four(4);
    LiteralInt two(2);
    BinaryExpr bexp(&four, BinaryOperator::addition, &two);
    EXPECT_EQ(dynamic_cast<LiteralInt *>(bexp.left)->value, 4);
}

TEST(BinaryExprTests, rightOperand) {
    LiteralInt four(4);
    LiteralInt two(2);
    BinaryExpr bexp(&four, BinaryOperator::addition, &two);
    EXPECT_EQ(dynamic_cast<LiteralInt *>(bexp.right)->value, 2);
}

TEST(BinaryExprTests, Operator) {
    LiteralInt four(4);
    LiteralInt two(2);
    BinaryExpr bexp(&four, BinaryOperator::addition, &two);
    EXPECT_EQ(bexp.op.op, BinaryOperator::addition);
}