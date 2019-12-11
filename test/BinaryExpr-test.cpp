#include <include/ast/LiteralInt.h>
#include <BinaryExpr.h>
#include "gtest/gtest.h"

TEST(BinaryExprTests, leftOperand) {
    LiteralInt four(4);
    LiteralInt two(2);
    BinaryExpr bexp(&four, BinaryOperator::addition, &two);
    EXPECT_EQ(dynamic_cast<LiteralInt *>(bexp.left)->getValue(), 4);
}

TEST(BinaryExprTests, rightOperand) {
    LiteralInt four(4);
    LiteralInt two(2);
    BinaryExpr bexp(&four, BinaryOperator::addition, &two);
    EXPECT_EQ(dynamic_cast<LiteralInt *>(bexp.right)->getValue(), 2);
}

TEST(BinaryExprTests, binaryOperator) {
    LiteralInt four(4);
    LiteralInt two(2);
    BinaryExpr bexp(&four, BinaryOperator::addition, &two);
    EXPECT_EQ(bexp.getOp(), BinaryOperator::addition);
}