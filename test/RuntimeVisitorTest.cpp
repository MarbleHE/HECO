#include <gtest/gtest.h>
#include <include/ast_opt/ast/Ast.h>
#include <include/ast_opt/visitor/CompileTimeExpressionSimplifier.h>
#include <include/ast_opt/visitor/RuntimeVisitor.h>
#include <include/ast_opt/visitor/PrintVisitor.h>
#include <random>
#include "AstTestingGenerator.h"

Matrix<int> *genRandomImageData(int imageSize, int numSlots) {
  // helpers to generate pseudorandom but reproducible numbers
  const unsigned int RANDOM_SEED = 874'332'410;
  std::mt19937 random_engine(RANDOM_SEED);
  std::uniform_int_distribution<int> distribution_1_1000(1, 1000);
  // generate a Matrix<int> of dimension (1, imageSize*imageSize) representing an image of size (imageSize, imageSize)
  std::vector<int> vec(numSlots);
  std::generate(vec.begin(), vec.begin() + (imageSize*imageSize), [&]() {
    return distribution_1_1000(random_engine);
  });
  return new Matrix<int>({vec});
}

std::vector<double> runLaplacianSharpeningFilter(Matrix<int> &img, int imgSize) {
  // initialize img2 as (1, imgSize*imgSize) matrix
  std::vector<std::vector<double>> weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  std::vector<double> img2(imgSize*imgSize);
  for (int x = 1; x < imgSize - 1; ++x) {
    for (int y = 1; y < imgSize - 1; ++y) {
      double value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
          value = value + weightMatrix[i + 1][j + 1]*img(0, imgSize*(x + i) + y + j);
        }
      }
      img2[imgSize*x + y] = img(0, imgSize*x + y) - (value/2);
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
  // use CTES to unroll the two innermost For-loops
  CompileTimeExpressionSimplifier ctes(CtesConfiguration(2));
  ctes.visit(ast);

  // a 32x32 image encoded as single 1'024 elements row vector
  auto imgData = genRandomImageData(32, 8192);
//  std::vector<int> data(1024);
//  std::iota(data.begin(), data.end(), 0);
//  Matrix<int> *imgData = new Matrix<int>({data});

  // execute the plaintext algorithm to know the expected result
  auto expectedResult = runLaplacianSharpeningFilter(*imgData, 32);
  Ciphertext ct = Ciphertext(expectedResult);

  // perform the actual execution by running the RuntimeVisitor
  RuntimeVisitor rt({{"img", new LiteralInt(imgData)}, {"imgSize", new LiteralInt(32)}});
  rt.visit(ast);

  // retrieve the RuntimeVisitor result
  auto retVal = rt.getReturnValues().front();
  // compare result with expected result (EXPECT_EQ)
  // TODO: Some of the indices are not equal.. this must be investigated further, not clear yet why.
  for (int i = 33; i < 33 + (30*30); ++i) {
    EXPECT_EQ(retVal->getElementAt(i), ct.getElementAt(i)) << " error for idx: " << i << std::endl;
  }
}
