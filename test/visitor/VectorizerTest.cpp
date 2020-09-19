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
    x *= {1,1,1,1,1,1,1,1,0};
    x += {0,0,0,0,0,0,0,0,5};
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST,*expectedAST));
}