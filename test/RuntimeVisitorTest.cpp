#include <gtest/gtest.h>
#include <include/ast_opt/ast/Ast.h>
#include <include/ast_opt/ast/Function.h>
#include <include/ast_opt/ast/FunctionParameter.h>
#include <include/ast_opt/ast/LogicalExpr.h>
#include <include/ast_opt/ast/VarAssignm.h>
#include <include/ast_opt/ast/VarDecl.h>
#include <include/ast_opt/ast/MatrixAssignm.h>
#include <include/ast_opt/ast/For.h>
#include <include/ast_opt/ast/Block.h>
#include <include/ast_opt/ast/Return.h>
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

TEST(RuntimeVisitorTests, DISABLED_rtCheckUsingCtes) { /* NOLINT */
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


std::vector<double> runLaplacianSharpeningFilterModified(Matrix<int> &img, int imgSize) {
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
      img2[imgSize*x + y] = 2*img(0, imgSize*x + y) - value;
    }
  }
  return img2;
}

TEST(RuntimeVisitorTests, rtCheckUsingExplicitAst) { /* NOLINT */
  // Executes the following AST using the RuntimeVisitor:
  // [BFV-compatible variant without division]
  // VecInt2D runLaplacianSharpeningAlgorithm(Vector<int> img, int imgSize) {
  //     Matrix<int> img2;
  //     for (int x = 1; x < imgSize - 1; ++x) {
  //         for (int y = 1; y < imgSize - 1; ++y) {
  //            img2[imgSize*x+y] = 2*img[imgSize*x+y] - (
  //                    + img[imgSize*(x-1)+y-1] + img[imgSize*x+y-1]        + img[imgSize*(x+1)+y-1]
  //                    + img[imgSize*(x-1)+y]   + img[imgSize*x+y] * (-8)   + img[imgSize*(x+1)+y]
  //                    + img[imgSize*(x-1)+y+1] + img[imgSize*x+y+1]        + img[imgSize*(x+1)+y+1] );
  //         }
  //     }
  //     return img2;
  // }


  // VecInt2D runLaplacianSharpeningAlgorithm(Vector<int> img, int imgSize, int x, int y) {
  auto func = new Function("runLaplacianSharpeningAlgorithm");
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("img")));
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, false), new Variable("imgSize")));

  func->addStatement(new VarDecl("img2", new Datatype(Types::INT)));

  // a helper to generate img[imgSize*(x-i)+y+j] terms
  auto createImgIdx = [](int i, int j) -> AbstractExpr * {
    auto buildTermI = [](int i) -> AbstractExpr * {
      if (i==0) {
        return new Variable("x");
      } else {
        return new OperatorExpr(new Operator(ADDITION), {new Variable("x"), new LiteralInt(i)});
      }
    };

    auto buildTermJ = [&](int j) -> AbstractExpr * {
      if (j==0) {
        return new OperatorExpr(new Operator(ADDITION),
                                {new OperatorExpr(new Operator(MULTIPLICATION),
                                                  {new Variable("imgSize"),
                                                   buildTermI(i)}),
                                 new Variable("y")});
      } else {
        return new OperatorExpr(new Operator(ADDITION),
                                {new OperatorExpr(new Operator(MULTIPLICATION),
                                                  {new Variable("imgSize"),
                                                   buildTermI(i)}),
                                 new Variable("y"),
                                 new LiteralInt(j)});
      }
    };
    return new MatrixElementRef(new Variable("img"), new LiteralInt(0), buildTermJ(j));
  };

  // img[imgSize*(x-1)+y-1]  * 1 + ... + img[imgSize*(x+1)+y+1]  * 1;
  auto varValue =
      new OperatorExpr(
          new Operator(ADDITION),
          {createImgIdx(-1, -1),
           createImgIdx(0, -1),
           createImgIdx(1, -1),
           createImgIdx(-1, 0),
           new OperatorExpr(new Operator(MULTIPLICATION), {createImgIdx(0, 0), new LiteralInt(-8)}),
           createImgIdx(1, 0),
           createImgIdx(-1, 1),
           createImgIdx(0, 1),
           createImgIdx(1, 1)});


  // img2[imgSize*x+y] = img[imgSize*x+y] - (value/2);
  auto secondLoopBody = new Block(
      new MatrixAssignm(new MatrixElementRef(new Variable("img2"),
                                             new LiteralInt(0),
                                             new OperatorExpr(
                                                 new Operator(ADDITION),
                                                 {new OperatorExpr(new Operator(MULTIPLICATION),
                                                                   {new Variable("imgSize"), new Variable("x")}),
                                                  new Variable("y")})),
                        new OperatorExpr(
                            new Operator(SUBTRACTION),
                            {new OperatorExpr(new Operator(MULTIPLICATION),
                                              {new MatrixElementRef(
                                                  new Variable("img"),
                                                  new LiteralInt(0),
                                                  new OperatorExpr(
                                                      new Operator(ADDITION),
                                                      {new OperatorExpr(new Operator(MULTIPLICATION),
                                                                        {new Variable("imgSize"), new Variable("x")}),
                                                       new Variable("y")})),
                                               new LiteralInt(2)}),
                             varValue})));

  // for (int y = 1; y < imgSize - 1; ++y)  -- 2nd level loop
  auto firstLoopBody = new Block(new For(new VarDecl("y", 1),
                                         new LogicalExpr(new Variable("y"),
                                                         SMALLER,
                                                         new ArithmeticExpr(new Variable("imgSize"), SUBTRACTION, 1)),
                                         new VarAssignm("y",
                                                        new ArithmeticExpr(new Variable("y"),
                                                                           ADDITION,
                                                                           new LiteralInt(1))),
                                         secondLoopBody));

  // for (int x = 1; x < imgSize - 1; ++x)  -- 1st level loop
  func->addStatement(new For(new VarDecl("x", 1),
                             new LogicalExpr(new Variable("x"),
                                             SMALLER,
                                             new ArithmeticExpr(new Variable("imgSize"), SUBTRACTION, 1)),
                             new VarAssignm("x",
                                            new ArithmeticExpr(new Variable("x"),
                                                               ADDITION,
                                                               new LiteralInt(1))),
                             firstLoopBody));

  // return img2;
  func->addStatement(new Return(new Variable("img2")));

  Ast ast;
  ast.setRootNode(func);

  // a 32x32 image encoded as single 1'024 elements row vector
  auto imgData = genRandomImageData(32, 8192);

  // execute the plaintext algorithm to know the expected result
  auto expectedResult = runLaplacianSharpeningFilterModified(*imgData, 32);
  Ciphertext ct = Ciphertext(expectedResult);

  // perform the actual execution by running the RuntimeVisitor
  RuntimeVisitor rt({{"img", new LiteralInt(imgData)}, {"imgSize", new LiteralInt(32)}});
  rt.visit(ast);

  // retrieve the RuntimeVisitor result
  auto retVal = rt.getReturnValues().front();
  std::vector<std::int64_t> vals = retVal->decryptAndDecode();

  // compare: our shadow plaintext computation vs. computations made on the SEAL ciphertext
  // FIXME: Some of the values are not equal.. this must be investigated further, not clear yet why.
  EXPECT_EQ(retVal->getNumCiphertextSlots(), vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    EXPECT_EQ(vals.at(i), retVal->getElementAt(i)) << "Plaintext result and ciphertext result mismatch!";
  }

  // compare: our shadown plaintext computation vs. reference implementation of Laplacian Sharpening algorithm
  // FIXME: Some of the values are not equal.. this must be investigated further, not clear yet why.
  for (int i = 33; i < 33 + (30*30); ++i) {
    EXPECT_EQ(retVal->getElementAt(i), ct.getElementAt(i)) << " error for idx: " << i << std::endl;
  }
}
