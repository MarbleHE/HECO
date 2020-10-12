#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/visitor/ExpressionBatcher.h"
#include "ast_opt/parser/Parser.h"
#include "../ASTComparison.h"
#include "gtest/gtest.h"
TEST(ExpressionBatcherTest, batchableExpression) {

  const char *inputChars = R""""(
    x = (a*b) + (c*d);
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ExpressionBatcher b;
  b.setRootScope(std::make_unique<Scope>(*inputAST));
  b.getRootScope().addIdentifier("x");
  b.getRootScope().addIdentifier("a");
  b.getRootScope().addIdentifier("b");
  b.getRootScope().addIdentifier("c");
  b.getRootScope().addIdentifier("d");

  auto assignment = dynamic_cast<Assignment*>(&*inputAST->begin());
  auto rhs = dynamic_cast<BinaryExpression&>(assignment->getValue());
  //auto cv =  b.batchExpression(rhs, BatchingConstraint());

  const char *expectedChars = R""""(
    __input0__ = __input0__ * __input1__;
    __input0__ = __input0__ + rotate(__input0__,1);
  )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  //TODO: How to compare ComplexValue?
  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
}

//TODO: Write lots of tests for batchability detection logic and think about algorithm shortcuts for "boring case" like sum.


TEST(ExpressionBatcherTest, cardioTestMegaExpression) {

  // Before Variable Substitution:
  //  risk = risk +++ (man && (age > 50));
  //  risk = risk +++ (woman && (age > 40));
  //  risk = risk +++ smoking;
  //  risk = risk +++ diabetic;
  //  risk = risk +++ high_blood_pressure;
  //  risk = risk +++ (40 > cholesterol);
  //  risk = risk +++ (weight > height);
  //  risk = risk +++ (30 > daily_physical_activity);
  //  risk = risk +++ (man && (alcohol > 3));
  //  risk = risk +++ (woman && (alcohol > 2));

  // With Variable Substitution
  // TODO: Here, variable substitution is somewhat less helpful.
  //   We need to implement expression-batching for this!
  // assuming risk = 0;
  //  risk =(man && (age > 50)) +++ (woman && (age > 40)) +++ smoking +++ diabetic
  //  +++ high_blood_pressure +++ (40 > cholesterol) +++ (weight > height) +++ (30 > daily_physical_activity)
  //  +++ (man && (alcohol > 3)) +++ (woman && (alcohol > 2));

  const char *inputChars = R""""(
   risk =(man && (age > 50)) +++ (woman && (age > 40)) +++ smoking +++ diabetic
   +++ high_blood_pressure +++ (40 > cholesterol) +++ (weight > height) +++ (30 > daily_physical_activity)
   +++ (man && (alcohol > 3)) +++ (woman && (alcohol > 2));
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  ExpressionBatcher b;
  b.setRootScope(std::make_unique<Scope>(*inputAST));
  b.getRootScope().addIdentifier("risk");
  b.getRootScope().addIdentifier("man");
  b.getRootScope().addIdentifier("age");
  b.getRootScope().addIdentifier("woman");
  b.getRootScope().addIdentifier("smoking");
  b.getRootScope().addIdentifier("diabetic");
  b.getRootScope().addIdentifier("high_blood_pressure");
  b.getRootScope().addIdentifier("cholesterol");
  b.getRootScope().addIdentifier("weight");
  b.getRootScope().addIdentifier("height");
  b.getRootScope().addIdentifier("daily_physical_activity");
  b.getRootScope().addIdentifier("alcohol");
  inputAST->accept(b);

  const char *expectedChars = R""""(
    risk = __input0__ *** (__input1__ > __input2__);
  )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
}