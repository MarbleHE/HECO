#include "gtest/gtest.h"
#include "ast_opt/ast/Operator.h"
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/ast/Variable.h"

class OperatorExprFixture : public ::testing::Test {
 protected:
  Operator *opAddition;
  Operator *opMultiplication;
  AbstractExpr *literalTwo;
  AbstractExpr *literalThree;

  OperatorExprFixture() {
    // operators
    opAddition = new Operator(ADDITION);
    opMultiplication = new Operator(MULTIPLICATION);
    // literal integers
    literalTwo = new LiteralInt(2);
    literalThree = new LiteralInt(3);
  }
};

TEST_F(OperatorExprFixture, defaultConstructor) {  /* NOLINT */
  // create operator expression
  OperatorExpr oe1(opAddition, {literalTwo, literalThree});

  // check operator
  EXPECT_TRUE(oe1.getOperator()->equals(opAddition->getOperatorSymbol()));

  // check operands
  auto operands = oe1.getOperands();
  EXPECT_TRUE(operands.at(0)->isEqual(new LiteralInt(5)));
}

TEST_F(OperatorExprFixture, singleArgumentConstructor) {  /* NOLINT */
  OperatorExpr oe(opAddition);

  // check operator
  EXPECT_TRUE(oe.getOperator()->equals(opAddition->getOperatorSymbol()));

  // check operands
  auto operands = oe.getOperands();
  EXPECT_EQ(oe.getOperands().size(), 0);
}

TEST_F(OperatorExprFixture, toStringRepresentation) {  /* NOLINT */
  // create operator expression
  OperatorExpr oe1(opAddition, {literalTwo, literalThree, new LiteralInt(332), new LiteralInt(11)});

  // print as string without children
  EXPECT_EQ(oe1.toString(false), std::string("OperatorExpr\n"));

  // print as string including children
  auto expectedStr = "OperatorExpr:\n"
                     "\tOperator (add)\n"
                     "\tLiteralInt (348)\n";
  EXPECT_EQ(oe1.toString(true), expectedStr);
}

TEST_F(OperatorExprFixture, setOperatorTest) {  /* NOLINT */
  // create operator expression
  OperatorExpr oe1(opAddition, {literalTwo, literalThree, new LiteralInt(332), new LiteralInt(11)});
  EXPECT_EQ(oe1.countChildrenNonNull(), 2); // opAddition and result of aggregation

  // change operator
  oe1.setOperator(opMultiplication);

  // check if change was successful
  EXPECT_EQ(oe1.countChildrenNonNull(), 2);
  EXPECT_TRUE(oe1.getOperator()->equals(opMultiplication->getOperatorSymbol()));
}

TEST_F(OperatorExprFixture, visitorSupport) {  /* NOLINT */
  // a test visitor
  struct TestVisitor : public Visitor {
    bool visitorVisitedOperatorExpr{false};

    void visit(OperatorExpr &elem) override {
      visitorVisitedOperatorExpr = true;
    }
  };

  TestVisitor tv;
  OperatorExpr(new Operator(ADDITION)).accept(tv);
  EXPECT_TRUE(tv.visitorVisitedOperatorExpr);
}

TEST_F(OperatorExprFixture, operatorTypeTest_arithmeticExprs) {  /* NOLINT */
  // arithmetic expressions
  OperatorExpr oeAdd = OperatorExpr(new Operator(ADDITION));
  EXPECT_TRUE(oeAdd.isArithmeticExpr());
  EXPECT_FALSE(oeAdd.isLogicalExpr());
  EXPECT_FALSE(oeAdd.isUnaryExpr());

  OperatorExpr oeSub = OperatorExpr(new Operator(SUBTRACTION));
  EXPECT_TRUE(oeSub.isArithmeticExpr());
  EXPECT_FALSE(oeSub.isLogicalExpr());
  EXPECT_FALSE(oeSub.isUnaryExpr());

  OperatorExpr oeMult = OperatorExpr(new Operator(MULTIPLICATION));
  EXPECT_TRUE(oeMult.isArithmeticExpr());
  EXPECT_FALSE(oeMult.isLogicalExpr());
  EXPECT_FALSE(oeMult.isUnaryExpr());

  OperatorExpr oeDiv = OperatorExpr(new Operator(DIVISION));
  EXPECT_TRUE(oeDiv.isArithmeticExpr());
  EXPECT_FALSE(oeDiv.isLogicalExpr());
  EXPECT_FALSE(oeDiv.isUnaryExpr());

  OperatorExpr oeMod = OperatorExpr(new Operator(MODULO));
  EXPECT_TRUE(oeMod.isArithmeticExpr());
  EXPECT_FALSE(oeMod.isLogicalExpr());
  EXPECT_FALSE(oeMod.isUnaryExpr());
}

TEST_F(OperatorExprFixture, operatorTypeTest_logicalExprs) {  /* NOLINT */
  // logical expressions
  OperatorExpr oeLAnd = OperatorExpr(new Operator(LOGICAL_AND));
  EXPECT_TRUE(oeLAnd.isLogicalExpr());
  EXPECT_FALSE(oeLAnd.isArithmeticExpr());
  EXPECT_FALSE(oeLAnd.isUnaryExpr());

  OperatorExpr oeLOr = OperatorExpr(new Operator(LOGICAL_OR));
  EXPECT_TRUE(oeLOr.isLogicalExpr());
  EXPECT_FALSE(oeLOr.isArithmeticExpr());
  EXPECT_FALSE(oeLOr.isUnaryExpr());

  OperatorExpr oeLXor = OperatorExpr(new Operator(LOGICAL_XOR));
  EXPECT_TRUE(oeLXor.isLogicalExpr());
  EXPECT_FALSE(oeLXor.isArithmeticExpr());
  EXPECT_FALSE(oeLXor.isUnaryExpr());

  OperatorExpr oeSmaller = OperatorExpr(new Operator(SMALLER));
  EXPECT_TRUE(oeSmaller.isLogicalExpr());
  EXPECT_FALSE(oeSmaller.isArithmeticExpr());
  EXPECT_FALSE(oeSmaller.isUnaryExpr());

  OperatorExpr oeSmallerEqual = OperatorExpr(new Operator(SMALLER_EQUAL));
  EXPECT_TRUE(oeSmallerEqual.isLogicalExpr());
  EXPECT_FALSE(oeSmallerEqual.isArithmeticExpr());
  EXPECT_FALSE(oeSmallerEqual.isUnaryExpr());

  OperatorExpr oeGreater = OperatorExpr(new Operator(GREATER));
  EXPECT_TRUE(oeGreater.isLogicalExpr());
  EXPECT_FALSE(oeGreater.isArithmeticExpr());
  EXPECT_FALSE(oeGreater.isUnaryExpr());

  OperatorExpr oeGreaterEqual = OperatorExpr(new Operator(GREATER_EQUAL));
  EXPECT_TRUE(oeGreaterEqual.isLogicalExpr());
  EXPECT_FALSE(oeGreaterEqual.isArithmeticExpr());
  EXPECT_FALSE(oeGreaterEqual.isUnaryExpr());

  OperatorExpr oeEqual = OperatorExpr(new Operator(EQUAL));
  EXPECT_TRUE(oeEqual.isLogicalExpr());
  EXPECT_FALSE(oeEqual.isArithmeticExpr());
  EXPECT_FALSE(oeEqual.isUnaryExpr());

  OperatorExpr oeUnequal = OperatorExpr(new Operator(UNEQUAL));
  EXPECT_TRUE(oeUnequal.isLogicalExpr());
  EXPECT_FALSE(oeUnequal.isArithmeticExpr());
  EXPECT_FALSE(oeUnequal.isUnaryExpr());
}

TEST_F(OperatorExprFixture, operatorTypeTest_unaryExprs) {  /* NOLINT */
  // unary expressions
  OperatorExpr oeNegation = OperatorExpr(new Operator(NEGATION));
  EXPECT_TRUE(oeNegation.isUnaryExpr());
  EXPECT_FALSE(oeNegation.isLogicalExpr());
  EXPECT_FALSE(oeNegation.isArithmeticExpr());
}

TEST_F(OperatorExprFixture, aggregatableOperatorAddition) {    /* NOLINT */
  OperatorExpr opExpr(new Operator(ADDITION), {new LiteralInt(0), new LiteralInt(9)});
  EXPECT_TRUE(opExpr.isEqual(new OperatorExpr(new Operator(ADDITION), {new LiteralInt(9)})));
}

TEST_F(OperatorExprFixture, partiallyAggregatableOperatorAddition) {    /* NOLINT */
  OperatorExpr opExpr(new Operator(ADDITION), {new LiteralInt(1), new Variable("x"), new LiteralInt(8)});
  EXPECT_TRUE(opExpr.isEqual(new OperatorExpr(new Operator(ADDITION), {new Variable("x"), new LiteralInt(9)})));
}

TEST_F(OperatorExprFixture, leftAssociativeOperatorDivision) {    /* NOLINT */
  OperatorExpr opExpr(new Operator(DIVISION),
                      {new LiteralInt(10), new LiteralInt(2), new Variable("x"), new LiteralInt(5)});
  EXPECT_TRUE(opExpr.isEqual(new OperatorExpr(new Operator(DIVISION),
                                              {new LiteralInt(5), new Variable("x"), new LiteralInt(5)})));
}

TEST_F(OperatorExprFixture, leftAssociateOperatorRequiringAllKnownOperands) {    /* NOLINT */
  OperatorExpr
      opExpr(new Operator(SMALLER), {new LiteralInt(1), new LiteralInt(3), new Variable("a"), new LiteralInt(2)});
  auto expectedOperatorExpr =
      new OperatorExpr(new Operator(SMALLER),
                       {new LiteralInt(1), new LiteralInt(3), new Variable("a"), new LiteralInt(2)});
  EXPECT_TRUE(opExpr.isEqual(expectedOperatorExpr));
}
