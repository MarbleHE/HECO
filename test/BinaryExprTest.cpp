#include "LiteralInt.h"
#include "BinaryExpr.h"
#include "gtest/gtest.h"

int getLiteralIntValue(AbstractExpr *abExpr) {
  return dynamic_cast<LiteralInt *>(abExpr)->getValue();
}

// TODO(pjattke): create tests for all possible combinations of operators and operands
// => 4 operand types * 2 combinations (lhs or rhs) * 14 binary operators + 4 unary operators
// = 116 Tests. Generate these tests somehow automatically?
// TODO(pjattke): after having tests, improve evaluation mechanism of BinaryExpr

TEST(BinaryExprTests, bexp3Add11) { /* NOLINT */
  int valLhs = 3, valRhs = 11;
  auto *b = new BinaryExpr(valLhs, OpSymb::BinaryOp::addition, valRhs);
  EXPECT_EQ(getLiteralIntValue(b->getLeft()), valLhs);
  EXPECT_EQ(getLiteralIntValue(b->getRight()), valRhs);
  EXPECT_EQ(b->getOp()->getOperatorString(), "add");
}

TEST(BinaryExprTests, bexp2Mult9Add3) { /* NOLINT */
  int valLeft = 2, valMid = 9, valRight = 3;
  auto *b = new BinaryExpr(valLeft, OpSymb::BinaryOp::multiplication,
                           new BinaryExpr(valMid, OpSymb::BinaryOp::addition, valRight));
  // lhs: outer BExp
  EXPECT_EQ(getLiteralIntValue(b->getLeft()), valLeft);
  EXPECT_EQ(b->getOp()->getOperatorString(), OpSymb::getTextRepr(OpSymb::BinaryOp::multiplication));

  // rhs: inner Bexp
  auto *r = dynamic_cast<BinaryExpr *>(b->getRight());
  EXPECT_EQ(getLiteralIntValue(r->getLeft()), valMid);
  EXPECT_EQ(getLiteralIntValue(r->getRight()), valRight);
  EXPECT_EQ(r->getOp()->getOperatorString(), OpSymb::getTextRepr(OpSymb::BinaryOp::addition));
}

