#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/visitor/Vectorizer.h"
#include "ast_opt/parser/Parser.h"
#include "../ASTComparison.h"
#include "gtest/gtest.h"

TEST(VectorizerTest, trivialVectors) {
  const char *inputChars = R""""(
    x[0] = y[0];
    x[1] = y[1];
    x[2] = y[2];
    x[3] = y[3];
    x[4] = y[4];
    x[5] = y[5];
    x[6] = y[6];
    x[7] = y[7];
    x[8] = y[8];
    x[9] = y[9];
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  inputAST->accept(v);

  const char *expectedChars = R""""(
    x = y;
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST,*expectedAST));
}

TEST(VectorizerTest, singleOutlierVector) {
  const char *inputChars = R""""(
    x[0] = y[0];
    x[1] = y[1];
    x[2] = y[2];
    x[3] = y[3];
    x[4] = y[4];
    x[5] = y[5];
    x[6] = y[6];
    x[7] = y[7];
    x[8] = y[8];
    x[9] = 5;
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  inputAST->accept(v);

  const char *expectedChars = R""""(
    x = y;
    x = x * {1,1,1,1,1,1,1,1,0};
    x = x + {0,0,0,0,0,0,0,0,5};
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST,*expectedAST));
}


TEST(VectorizerTest, sumStatementsPowerOfTwo) {
  //If sum is vector valued, this would mean something very different
  // Specifically, it would mean that in each step, x[i] is added to each slot.
  // TODO: Therefore, we need to pass in some additional information to the Vectorizer
  // i.e. that sum has a scalar datatype.
  //TODO: We *do* need to differentiate between vector and scalar types in our input language and during compilation!
  const char *inputChars = R""""(
    sum = sum + x[0];
    sum = sum + x[1];
    sum = sum + x[2];
    sum = sum + x[3];
    sum = sum + x[4];
    sum = sum + x[5];
    sum = sum + x[6];
    sum = sum + x[7];
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  inputAST->accept(v);

  //TODO: How to communicate the slot of the result to the runtime system?
  // maybe something like __sum__i__ where i is the slot index?
  // Then we could get rid of the sum = sum * {1,0,0...} masking + it would be less brittle
  const char *expectedChars = R""""(
    sum = x + rotate(x, 4);
    sum = sum + rotate(sum, 2);
    sum = sum + rotate(sum, 1);
    sum = sum * {1,0,0,0,0,0,0,0};
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST,*expectedAST));
}

TEST(VectorizerTest, sumStatementsGeneral) {
  const char *inputChars = R""""(
    sum = sum + x[0];
    sum = sum + x[1];
    sum = sum + x[2];
    sum = sum + x[3];
    sum = sum + x[4];
    sum = sum + x[5];
    sum = sum + x[6];
    sum = sum + x[7];
    sum = sum + x[8];
    sum = sum + x[9];
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  inputAST->accept(v);

  // First extend the vector to the next power of two (and mask away any potential garbage)
  // TODO: Is there a away to avoid this by keeping some additional info or having invariants/guarantees on inputs?
  const char *expectedChars = R""""(
    sum = x * {1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0};
    sum = sum + rotate(sum, 8);
    sum = sum + rotate(sum, 4);
    sum = sum + rotate(sum, 2);
    sum = sum + rotate(sum, 1);
    sum = sum * {1,0,0,0,0,0,0,0};
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST,*expectedAST));
}