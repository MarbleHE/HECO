#include <gtest/gtest.h>
#include <include/ast_opt/ast/Ast.h>
#include <include/ast_opt/visitor/CompileTimeExpressionSimplifier.h>
#include <include/ast_opt/visitor/RuntimeVisitor.h>
#include <include/ast_opt/visitor/PrintVisitor.h>
#include <random>
#include "AstTestingGenerator.h"

Matrix<int> *genRandomImageData(int imageSize) {
  // helpers to generate pseudorandom reproducible numbers
  const unsigned int RANDOM_SEED = 874'332'410;
  std::mt19937 random_engine(RANDOM_SEED);
  std::uniform_int_distribution<int> distribution_1_100(1, 100);
  // generate a Matrix<int> of dimension (1, imageSize*imageSize) representing an image of size (imageSize, imageSize)
  std::vector<int> vec(imageSize);
  std::generate(vec.begin(), vec.end(), [&]() {
    return distribution_1_100(random_engine);
  });
  return new Matrix<int>({vec});
}

Matrix<int> runLaplacianSharpeningFilter(Matrix<int> &img, int imgSize) {
  // initialize img2 as (1, imgSize*imgSize) matrix
  Matrix<int> img2({1, std::vector<int>(imgSize*imgSize)});
  for (int x = 1; x < imgSize - 1; ++x) {
    for (int y = 1; y < imgSize - 1; ++y) {
      img2(0, imgSize*x + y) =
          img(0, imgSize*x + y)
              - (img(0, imgSize*(x - 1) + y - 1)
                  + img(0, imgSize*x + y - 1)
                  + img(0, imgSize*(x + 1) + y - 1)
                  + img(0, imgSize*(x - 1) + y)
                  + img(0, imgSize*x + y)*(-8)
                  + img(0, imgSize*(x + 1) + y)
                  + img(0, imgSize*(x - 1) + y + 1)
                  + img(0, imgSize*x + y + 1)
                  + img(0, imgSize*(x + 1) + y + 1))/2;
    }
  }
  return img2;
}

TEST(RuntimeVisitorTests, rtCheck) { /* NOLINT */
  // create the Laplacian sharpening filter AST
  Ast ast;
  AstTestingGenerator::generateAst(60, ast);

  /* execute the CTES resulting in the following simplified AST:
    VecInt2D runLaplacianSharpeningAlgorithm(Vector<int> img, int imgSize) {
        Matrix<int> img2;
        for (int x = 1; x < imgSize - 1; ++x) {
            for (int y = 1; y < imgSize - 1; ++y) {
                img2[imgSize*x+y] = img[imgSize*x+y]
                    - (+ img[imgSize*(x-1)+y-1] + img[imgSize*x+y-1]        + img[imgSize*(x+1)+y-1]
                       + img[imgSize*(x-1)+y]   + img[imgSize*x+y] * (-8)   + img[imgSize*(x+1)+y]
                       + img[imgSize*(x-1)+y+1] + img[imgSize*x+y+1]        + img[imgSize*(x+1)+y+1] ) / 2;
            }
        }
        return img2;
    }
  */
  CompileTimeExpressionSimplifier ctes(CtesConfiguration(2));
  ctes.visit(ast);

  // a 32x32 image encoded as single 1'024 elements row vector
  auto imgData = genRandomImageData(32);

  // execute the plaintext algorithm to know the expected result
  auto expectedResult = runLaplacianSharpeningFilter(*imgData, 32);

  // perform the actual execution by running the RuntimeVisitor
  RuntimeVisitor rt({{"img", new LiteralInt(imgData)}, {"imgSize", new LiteralInt(32)}});
  rt.visit(ast);

  // TODO: retrieve the RuntimeVisitor result
//  rt.getResult();

  // TODO: compare result with expected result (EXPECT_EQ)
//  EXPECT_EQ(expectedResult, rtvResult);
}
