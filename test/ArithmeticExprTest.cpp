#include "LiteralInt.h"
#include "ArithmeticExpr.h"
#include "gtest/gtest.h"

int getLiteralIntValue(AbstractExpr *aaexpr) {
  return dynamic_cast<LiteralInt *>(aaexpr)->getValue();
}

// TODO(pjattke): create tests for all possible combinations of operators and operands
// => 4 operand Types * 2 combinations (lhs or rhs) * 14 binary operators + 4 unary operators
// = 116 Tests. Generate these tests somehow automatically?
// TODO(pjattke): after having tests, improve evaluation mechanism of ArithmeticExpr

TEST(ArithmeticExprTest, aexp3Add11) { /* NOLINT */
  int valLhs = 3, valRhs = 11;
  auto *b = new ArithmeticExpr(valLhs, ArithmeticOp::addition, valRhs);
  EXPECT_EQ(getLiteralIntValue(b->getLeft()), valLhs);
  EXPECT_EQ(getLiteralIntValue(b->getRight()), valRhs);
  EXPECT_EQ(b->getOp()->getOperatorString(), "add");
}

TEST(ArithmeticExprTest, aexp2Mult9Add3) { /* NOLINT */
  int valLeft = 2, valMid = 9, valRight = 3;
  auto *b = new ArithmeticExpr(valLeft, ArithmeticOp::multiplication,
                               new ArithmeticExpr(valMid, ArithmeticOp::addition, valRight));
  // lhs: outer Aexp
  EXPECT_EQ(getLiteralIntValue(b->getLeft()), valLeft);
  EXPECT_EQ(b->getOp()->getOperatorString(), OpSymb::getTextRepr(ArithmeticOp::multiplication));

  // rhs: inner Aexp
  auto *r = dynamic_cast<ArithmeticExpr *>(b->getRight());
  EXPECT_EQ(getLiteralIntValue(r->getLeft()), valMid);
  EXPECT_EQ(getLiteralIntValue(r->getRight()), valRight);
  EXPECT_EQ(r->getOp()->getOperatorString(), OpSymb::getTextRepr(ArithmeticOp::addition));
}

