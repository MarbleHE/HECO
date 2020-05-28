#include "ast_opt/evaluation/EvaluationAlgorithms.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <functional>
#include <ast_opt/ast/Ast.h>
#include <ast_opt/ast/Function.h>
#include <ast_opt/ast/GetMatrixSize.h>
#include <ast_opt/ast/Datatype.h>
#include <ast_opt/ast/Variable.h>
#include <ast_opt/ast/LogicalExpr.h>
#include <ast_opt/ast/For.h>
#include <ast_opt/ast/VarDecl.h>
#include <ast_opt/ast/VarAssignm.h>
#include <ast_opt/ast/ArithmeticExpr.h>
#include <ast_opt/ast/If.h>
#include <ast_opt/ast/MatrixElementRef.h>
#include <ast_opt/ast/MatrixAssignm.h>
#include <ast_opt/ast/OperatorExpr.h>
#include <ast_opt/ast/Block.h>
#include <ast_opt/ast/Variable.h>
#include <ast_opt/ast/Return.h>
#include <ast_opt/ast/Rotate.h>
#include <ast_opt/mockup_classes/Ciphertext.h>

typedef std::vector<std::vector<int>> VecInt2D;

// Credits to Hariom Singh from stackoverflow.com (https://stackoverflow.com/a/45577531/3017719).
std::vector<int> cyclicRotate(std::vector<int> &A, unsigned long rotationFactor) {
  if (A.empty() || A.size()==1) return A;
  rotationFactor = rotationFactor%A.size();
  if (rotationFactor==0) return A;
  std::rotate(A.begin(), A.begin() + rotationFactor, A.end());
  return A;
}

std::pair<float, float> EvaluationAlgorithms::runLinearRegression(std::vector<std::pair<float, float>> datapoints) {
  int numDatapoints = datapoints.size();
  float sumX = 0, sumXX = 0, sumY = 0, sumXY = 0;
  for (int i = 0; i < numDatapoints; i++) {
    sumX = sumX + datapoints.at(i).first;
    sumXX = sumXX + datapoints.at(i).first*datapoints.at(i).first;
    sumY = sumY + datapoints.at(i).second;
    sumXY = sumXY + datapoints.at(i).first*datapoints.at(i).second;
  }

  // calculate regression parameters a, b
  float b = (static_cast<float>(numDatapoints)*sumXY - sumX*sumY)/(static_cast<float>(numDatapoints)*sumXX - sumX*sumX);
  float a = (sumY - b*sumX)/static_cast<float>(numDatapoints);

  // display result and equation of regression line (y = ax + bx)
//  std::cout << "[Result] a: " << a << ", b: " << b << std::endl;
//  std::cout << "Equation of best fit: y = " << a << " + " << b << "x";

  return std::make_pair(a, b);
}

void EvaluationAlgorithms::genLinearRegressionAst(Ast &ast) {
  // runLinearRegression(Vector<secret_int> dataX, Vector<secret_int> dataY) {
  //    int N = dataX.size();
  //    int sumX = 0;
  //    int sumXX = 0;
  //    int sumY = 0;
  //    int sumXY = 0;
  //    for (int i = 0; i < N; i = i + 1) {
  //       sumX = sumX + dataX[i];
  //       sumXX = sumXX + dataX[i] * dataX[i];
  //       sumY = sumY + dataY[i];
  //       sumXY = sumXY + dataX[i] * dataY[i];
  //    }
  //    return sumX, sumXX, sumY, sumXY;
  // }
  //
  // Client computes regression parameters ax+b as follow:
  //   int b = (N*sumXY - sumX*sumY) / (N*sumXX - sumX*sumX);
  //   int a = (sumY - b*sumX)/N;
  auto func = new Function("runLinearRegression");

  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("dataX")));
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("dataY")));

  func->addStatement(new VarDecl("N",
                                 new Datatype(Types::INT),
                                 new GetMatrixSize(new Variable("dataX"), new LiteralInt(1))));
  func->addStatement(new VarDecl("sumX", 0));
  func->addStatement(new VarDecl("sumXX", 0));
  func->addStatement(new VarDecl("sumY", 0));
  func->addStatement(new VarDecl("sumXY", 0));

  auto *forLoopBlock = new Block();
  forLoopBlock->addChild(new VarAssignm("sumX",
                                        new OperatorExpr(new Operator(ADDITION),
                                                         {new Variable("sumX"),
                                                          new MatrixElementRef(new Variable("dataX"),
                                                                               new LiteralInt(0),
                                                                               new Variable("i"))})));
  forLoopBlock->addChild(new VarAssignm("sumXX",
                                        new OperatorExpr(new Operator(ADDITION),
                                                         {new Variable("sumXX"),
                                                          new OperatorExpr(new Operator(MULTIPLICATION),
                                                                           {new MatrixElementRef(new Variable("dataX"),
                                                                                                 new LiteralInt(0),
                                                                                                 new Variable("i")),
                                                                            new MatrixElementRef(new Variable("dataX"),
                                                                                                 new LiteralInt(0),
                                                                                                 new Variable("i"))})})));
  forLoopBlock->addChild(new VarAssignm("sumY",
                                        new OperatorExpr(new Operator(ADDITION),
                                                         {new Variable("sumY"),
                                                          new MatrixElementRef(new Variable("dataY"),
                                                                               new LiteralInt(0),
                                                                               new Variable("i"))})));
  forLoopBlock->addChild(new VarAssignm("sumXY",
                                        new OperatorExpr(new Operator(ADDITION),
                                                         {new Variable("sumXY"),
                                                          new OperatorExpr(new Operator(MULTIPLICATION),
                                                                           {new MatrixElementRef(new Variable("dataX"),
                                                                                                 new LiteralInt(0),
                                                                                                 new Variable("i")),
                                                                            new MatrixElementRef(new Variable("dataY"),
                                                                                                 new LiteralInt(0),
                                                                                                 new Variable("i"))})})));

  func->addStatement(new For(new VarDecl("i", 0),
                             new LogicalExpr(new Variable("i"), SMALLER, new Variable("N")),
                             new VarAssignm("i",
                                            new OperatorExpr(new Operator(ADDITION),
                                                             {new Variable("i"), new LiteralInt(1)})),
                             forLoopBlock));

  func->addStatement(new Return({new Variable("sumX"), new Variable("sumXX"), new Variable("sumY"),
                                 new Variable("sumXY")}));

  ast.setRootNode(func);
}

void EvaluationAlgorithms::genLinearRegressionAstAfterCtes(Ast &ast) {
  // runLinearRegression(Vector<int> dataX, Vector<int> dataY) {
  //    int sumX = 0;
  //    int sumXX = 0;
  //    int sumY = 0;
  //    int sumXY = 0;
  //    {
  //        int i = 0;
  //        for (; i < dataX.size() && i+1 < dataX.size() && i+2 < dataX.size(); ) {
  //           // i, i+1, i+2
  //           sumX = sumX + dataX[i] + dataX[i+1] + dataX[i+2];
  //           sumXX = sumXX + dataX[i] * dataX[i] + dataX[i+1] * dataX[i+1] + dataX[i+2] * dataX[i+2];
  //           sumY = sumY + dataY[i] + dataY[i+1] + dataY[i+2];
  //           sumXY = sumXY + dataX[i] * dataY[i] + dataX[i+1] * dataY[i+1] + dataX[i+2] * dataY[i+2];
  //           i = i + 3;
  //        }
  //        for (; i < N; i++) {
  //           sumX = sumX + dataX[i];
  //           sumXX = sumXX + dataX[i] * dataX[i];
  //           sumY = sumY + dataY[i];
  //           sumXY = sumXY + dataX[i] * dataY[i];
  //        }
  //    }
  //    return sumX, sumXX, sumY, sumXY;
  // }
  auto func = new Function("runLinearRegression");
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("dataX")));
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("dataY")));
  func->addStatement(new VarDecl("sumX", 0));
  func->addStatement(new VarDecl("sumXX", 0));
  func->addStatement(new VarDecl("sumY", 0));
  func->addStatement(new VarDecl("sumXY", 0));

  auto unrolledLoopsBlock = new Block();
  unrolledLoopsBlock->addChild(new VarDecl("i", 0));

  auto forLoopBlock = new Block();

  // i, i+1, i+2
  forLoopBlock->addChild(
      new VarAssignm("sumX", new OperatorExpr(new Operator(ADDITION),
                                              {new Variable("sumX"),
                                               new MatrixElementRef(new Variable("dataX"),
                                                                    new LiteralInt(0),
                                                                    new Variable("i")),
                                               new MatrixElementRef(new Variable("dataX"),
                                                                    new LiteralInt(0),
                                                                    new OperatorExpr(new Operator(ADDITION),
                                                                                     {new Variable("i"),
                                                                                      new LiteralInt(1)})),
                                               new MatrixElementRef(new Variable("dataX"),
                                                                    new LiteralInt(0),
                                                                    new OperatorExpr(new Operator(ADDITION),
                                                                                     {new Variable("i"),
                                                                                      new LiteralInt(2)}))})));
  forLoopBlock->addChild(
      new VarAssignm("sumXX",
                     new OperatorExpr(new Operator(ADDITION),
                                      {new Variable("sumXX"),
                                       new OperatorExpr(new Operator(MULTIPLICATION),
                                                        {new MatrixElementRef(new Variable("dataX"),
                                                                              new LiteralInt(0),
                                                                              new Variable("i")),
                                                         new MatrixElementRef(new Variable("dataX"),
                                                                              new LiteralInt(0),
                                                                              new Variable("i"))
                                                        }),
                                       new OperatorExpr(new Operator(MULTIPLICATION),
                                                        {new MatrixElementRef(new Variable("dataX"),
                                                                              new LiteralInt(0),
                                                                              new OperatorExpr(new Operator(ADDITION),
                                                                                               {new Variable("i"),
                                                                                                new LiteralInt(1)})),
                                                         new MatrixElementRef(new Variable("dataX"),
                                                                              new LiteralInt(0),
                                                                              new OperatorExpr(new Operator(ADDITION),
                                                                                               {new Variable("i"),
                                                                                                new LiteralInt(1)}))
                                                        }),
                                       new OperatorExpr(new Operator(MULTIPLICATION),
                                                        {new MatrixElementRef(new Variable("dataX"),
                                                                              new LiteralInt(0),
                                                                              new OperatorExpr(new Operator(ADDITION),
                                                                                               {new Variable("i"),
                                                                                                new LiteralInt(2)})),
                                                         new MatrixElementRef(new Variable("dataX"),
                                                                              new LiteralInt(0),
                                                                              new OperatorExpr(new Operator(ADDITION),
                                                                                               {new Variable("i"),
                                                                                                new LiteralInt(2)}))
                                                        })})));
  forLoopBlock->addChild(new VarAssignm("sumY",
                                        new OperatorExpr(new Operator(ADDITION),
                                                         {new Variable("sumY"),
                                                          new MatrixElementRef(new Variable("dataY"),
                                                                               new LiteralInt(0),
                                                                               new Variable("i")),
                                                          new MatrixElementRef(new Variable("dataY"),
                                                                               new LiteralInt(0),
                                                                               new OperatorExpr(new Operator(ADDITION),
                                                                                                {new Variable("i"),
                                                                                                 new LiteralInt(1)})),
                                                          new MatrixElementRef(new Variable("dataY"),
                                                                               new LiteralInt(0),
                                                                               new OperatorExpr(new Operator(ADDITION),
                                                                                                {new Variable("i"),
                                                                                                 new LiteralInt(2)}))
                                                         })));
  forLoopBlock->addChild(new VarAssignm("sumXY",
                                        new OperatorExpr(new Operator(ADDITION),
                                                         {new Variable("sumXY"),
                                                          new OperatorExpr(new Operator(MULTIPLICATION),
                                                                           {new MatrixElementRef(
                                                                               new Variable("dataX"),
                                                                               new LiteralInt(0),
                                                                               new Variable("i")),
                                                                            new MatrixElementRef(
                                                                                new Variable("dataY"),
                                                                                new LiteralInt(0),
                                                                                new Variable("i")
                                                                            )}),
                                                          new OperatorExpr(new Operator(MULTIPLICATION),
                                                                           {new MatrixElementRef(
                                                                               new Variable("dataX"),
                                                                               new LiteralInt(0),
                                                                               new OperatorExpr(new Operator(ADDITION),
                                                                                                {new Variable("i"),
                                                                                                 new LiteralInt(1)})),
                                                                            new MatrixElementRef(
                                                                                new Variable("dataY"),
                                                                                new LiteralInt(0),
                                                                                new OperatorExpr(new Operator(ADDITION),
                                                                                                 {new Variable("i"),
                                                                                                  new LiteralInt(1)})
                                                                            )}),
                                                          new OperatorExpr(new Operator(MULTIPLICATION),
                                                                           {new MatrixElementRef(
                                                                               new Variable("dataX"),
                                                                               new LiteralInt(0),
                                                                               new OperatorExpr(new Operator(ADDITION),
                                                                                                {new Variable("i"),
                                                                                                 new LiteralInt(2)})),
                                                                            new MatrixElementRef(
                                                                                new Variable("dataY"),
                                                                                new LiteralInt(0),
                                                                                new OperatorExpr(new Operator(ADDITION),
                                                                                                 {new Variable("i"),
                                                                                                  new LiteralInt(2)})
                                                                            )})})));

  forLoopBlock->addChild(new VarAssignm("i",
                                        new OperatorExpr(new Operator(ADDITION),
                                                         {new Variable("i"), new LiteralInt(3)})));

  unrolledLoopsBlock->addChild(new For(nullptr,
                                       new OperatorExpr(new Operator(LOGICAL_AND), {
                                           new OperatorExpr(new Operator(SMALLER),
                                                            {new Variable("i"),
                                                             new GetMatrixSize(new Variable("dataX"),
                                                                               new LiteralInt(1))}),
                                           new OperatorExpr(new Operator(SMALLER),
                                                            {new OperatorExpr(new Operator(ADDITION),
                                                                              {new Variable("i"),
                                                                               new LiteralInt(1)}),
                                                             new GetMatrixSize(new Variable("dataX"),
                                                                               new LiteralInt(1))}),
                                           new OperatorExpr(new Operator(SMALLER),
                                                            {new OperatorExpr(new Operator(ADDITION),
                                                                              {new Variable("i"),
                                                                               new LiteralInt(2)}),
                                                             new GetMatrixSize(new Variable("dataX"),
                                                                               new LiteralInt(1))})
                                       }),
                                       nullptr,
                                       forLoopBlock));

  // == CLEANUP LOOP ===
  auto cleanupLoopBlock = new Block();
  cleanupLoopBlock->addChild(new VarAssignm("sumX",
                                            new ArithmeticExpr(new Variable("sumX"), ADDITION, new MatrixElementRef
                                                (new Variable("dataX"), new LiteralInt(0), new Variable("i")))));
  cleanupLoopBlock->addChild(new VarAssignm("sumXX",
                                            new ArithmeticExpr(new Variable("sumXX"), ADDITION,
                                                               new ArithmeticExpr(new MatrixElementRef(new Variable(
                                                                   "dataX"),
                                                                                                       new LiteralInt(0),
                                                                                                       new Variable("i")),
                                                                                  MULTIPLICATION,
                                                                                  new MatrixElementRef(new Variable(
                                                                                      "dataX"),
                                                                                                       new LiteralInt(0),
                                                                                                       new Variable("i"))
                                                               ))));
  cleanupLoopBlock->addChild(new VarAssignm("sumY",
                                            new ArithmeticExpr(new Variable("sumY"), ADDITION, new MatrixElementRef
                                                (new Variable("dataY"), new LiteralInt(0), new Variable("i")))));
  cleanupLoopBlock->addChild(new VarAssignm("sumXY",
                                            new ArithmeticExpr(new Variable("sumXY"), ADDITION,
                                                               new ArithmeticExpr(new MatrixElementRef(new Variable(
                                                                   "dataX"),
                                                                                                       new LiteralInt(0),
                                                                                                       new Variable("i")),
                                                                                  MULTIPLICATION,
                                                                                  new MatrixElementRef(new Variable(
                                                                                      "dataY"),
                                                                                                       new LiteralInt(0),
                                                                                                       new Variable("i"))
                                                               ))));
  unrolledLoopsBlock->addChild(new For(nullptr,
                                       new LogicalExpr(new Variable("i"),
                                                       SMALLER,
                                                       new GetMatrixSize(new Variable("dataX"), new LiteralInt(1))),
                                       new VarAssignm("i",
                                                      new ArithmeticExpr(new Variable("i"),
                                                                         ADDITION,
                                                                         new LiteralInt(1))),
                                       cleanupLoopBlock));

  func->addStatement(unrolledLoopsBlock);

  // RETURN STMT
  func->addStatement(new Return({new Variable("sumX"), new Variable("sumXX"),
                                 new Variable("sumY"), new Variable("sumXY")}));
  ast.setRootNode(func);
}

void EvaluationAlgorithms::runPolynomialRegression(const std::vector<int> &x, const std::vector<int> &y) {
  // N := x.size()
  std::vector<int> r(x.size(), 0);
  // fill r with sequentially increasing numbers 0, 1, 2, ..., x.size()
  std::iota(r.begin(), r.end(), 0);
  double meanX = std::accumulate(x.begin(), x.end(), 0.0)/x.size();
  double meanY = std::accumulate(y.begin(), y.end(), 0.0)/y.size();
  double x2m = 0.0, x3m = 0.0, x4m = 0.0;
  for (int i = 0; i < r.size(); ++i) {
    x2m += r.at(i)*r.at(i);
    x3m += r.at(i)*r.at(i)*r.at(i);
    x4m += r.at(i)*r.at(i)*r.at(i)*r.at(i);
  }
  // x2m = (∑ x_i * x_i) / N
  x2m /= r.size();
  // x3m = (∑ x_i * x_i * x_i) / N
  x3m /= r.size();
  // x4m = (∑ x_i * x_i * x_i * x_i) / N
  x4m /= r.size();

  // computes ( 0.0 + (x_1*y_1) + (x_2*y_2) + ... + (x_N + y_N) ) / N
  double xym
      = std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<double>{}, std::multiplies<double>{});
  xym /= x.size() < y.size() ? x.size() : y.size();

  double x2ym = std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0,
                                      std::plus<double>{}, [](double a, double b) { return a*a*b; });
  x2ym /= x.size() < y.size() ? x.size() : y.size();

  // compute parameters a,b,c of trend line y = a + bx + cx^{2}
  double sxx = x2m - meanX*meanX;
  double sxy = xym - meanX*meanY;
  double sxx2 = x3m - meanX*x2m;
  double sx2x2 = x4m - x2m*x2m;
  double sx2y = x2ym - x2m*meanY;

  double b = (sxy*sx2x2 - sx2y*sxx2)/(sxx*sx2x2 - sxx2*sxx2);
  double c = (sx2y*sxx - sxy*sxx2)/(sxx*sx2x2 - sxx2*sxx2);
  double a = meanY - b*meanX - c*x2m;

  // prints the computed result
  auto abc = [a, b, c](int xx) { return a + b*xx + c*xx*xx; };
//  std::cout << "y = " << a << " + " << b << "x + " << c << "x^2" << std::endl;
//  std::cout << " Input  Approximation" << std::endl;
//  std::cout << " x   y     y1" << std::endl;
  auto xit = x.cbegin();
  auto yit = y.cbegin();
  while (xit!=x.cend() && yit!=y.cend()) {
    printf("%2d %3d  %5.1f\n", *xit, *yit, abc(*xit));
    xit = std::next(xit);
    yit = std::next(yit);
  }
}

void EvaluationAlgorithms::genPolynomialRegressionAst(Ast &ast) {
  // Computes sums xym and x2ym, see runPolynomialRegression for full code including client computations.
  //
  // computePolynomialRegressionSums(Vector<secret_int> x, Vector<secret_int> y) {
  //   double xym = 0;
  //   double x2ym = 0;
  //   for (int i = 0; i < x.size(); ++i) {
  //     xym = xym + x[i]*x[i];
  //     x2ym = x2ym + x[i]*x[i]*y[i];
  //   }
  //   return xym, x2ym;
  // }

  auto func = new Function("computePolynomialRegressionSums");
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("x")));
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("y")));

  func->addStatement(new VarDecl("xym", 0));
  func->addStatement(new VarDecl("x2ym", 0));

  auto forLoopBlock = new Block();

  forLoopBlock->addChild(
      new VarAssignm("xym", new OperatorExpr(new Operator(ADDITION),
                                             {new Variable("xym"),
                                              new OperatorExpr(new Operator(MULTIPLICATION),
                                                               {new MatrixElementRef(
                                                                   new Variable("x"),
                                                                   new LiteralInt(0),
                                                                   new Variable("i")),
                                                                new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new Variable("i"))})})));

  forLoopBlock->addChild(
      new VarAssignm("x2ym", new OperatorExpr(new Operator(ADDITION),
                                              {new Variable("x2ym"),
                                               new OperatorExpr(new Operator(MULTIPLICATION),
                                                                {new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new Variable("i")),
                                                                 new MatrixElementRef(
                                                                     new Variable("x"),
                                                                     new LiteralInt(0),
                                                                     new Variable("i")),
                                                                 new MatrixElementRef(
                                                                     new Variable("y"),
                                                                     new LiteralInt(0),
                                                                     new Variable("i"))})})));

  func->addStatement(new For(new VarDecl("i", 0),
                             new OperatorExpr(new Operator(SMALLER), {new Variable("i"),
                                                                      new GetMatrixSize(new Variable("x"),
                                                                                        new LiteralInt(1))}),
                             new VarAssignm("i", new OperatorExpr(new Operator(ADDITION),
                                                                  {new Variable("i"), new LiteralInt(1)})),
                             forLoopBlock));

  func->addStatement(new Return({new Variable("xym"), new Variable("x2ym")}));

  ast.setRootNode(func);
}

void EvaluationAlgorithms::genPolynomialRegressionAstAfterCtes(Ast &ast) {
  // Computes sums xym and x2ym, see runPolynomialRegression for full code including client computations.
  //
  // computePolynomialRegressionSums(Vector<secret_int> x, Vector<secret_int> y) {
  //   double xym = 0;
  //   double x2ym = 0;
  //   {
  //      int i = 0;
  //      for (; i < x.size() && i+1 < x.size() && i+2 < x.size();) {
  //         xym = xym + x[i]*x[i] + x[i+1]*x[i+1] + x[i+2]*x[i+2];
  //         x2ym = x2ym + x[i]*x[i]*y[i] + x[i+1]*x[i+1]*y[i+1] + x[i+2]*x[i+2]*y[i+2];
  //         i = i+3;
  //      }
  //      for (; i < x.size(); i = i+1) {
  //         xym = xym + x[i]*x[i];
  //         x2ym = x2ym + x[i]*x[i]*y[i];
  //     }
  //   }
  //   return xym, x2ym;
  // }

  auto func = new Function("computePolynomialRegressionSums");
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("x")));
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("y")));

  func->addStatement(new VarDecl("xym", 0));
  func->addStatement(new VarDecl("x2ym", 0));

  auto unrolledLoopsBlock = new Block();
  func->addStatement(unrolledLoopsBlock);

  // UNROLLED LOOP
  unrolledLoopsBlock->addChild(new VarDecl("i", 0));
  auto unwindedForLoopBlock = new Block();
  unwindedForLoopBlock->addChild(
      new VarAssignm("xym", new OperatorExpr(new Operator(ADDITION),
                                             {new Variable("xym"),
                                              new OperatorExpr(new Operator(MULTIPLICATION),
                                                               {new MatrixElementRef(
                                                                   new Variable("x"),
                                                                   new LiteralInt(0),
                                                                   new Variable("i")),
                                                                new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new Variable("i"))}),
                                              new OperatorExpr(new Operator(MULTIPLICATION),
                                                               {new MatrixElementRef(
                                                                   new Variable("x"),
                                                                   new LiteralInt(0),
                                                                   new OperatorExpr(new Operator(ADDITION),
                                                                                    {new Variable("i"),
                                                                                     new LiteralInt(1)})),
                                                                new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new OperatorExpr(new Operator(ADDITION),
                                                                                     {new Variable("i"),
                                                                                      new LiteralInt(1)}))}),
                                              new OperatorExpr(new Operator(MULTIPLICATION),
                                                               {new MatrixElementRef(
                                                                   new Variable("x"),
                                                                   new LiteralInt(0),
                                                                   new OperatorExpr(new Operator(ADDITION),
                                                                                    {new Variable("i"),
                                                                                     new LiteralInt(2)})),
                                                                new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new OperatorExpr(new Operator(ADDITION),
                                                                                     {new Variable("i"),
                                                                                      new LiteralInt(2)}))})})));

  unwindedForLoopBlock->addChild(
      new VarAssignm("x2ym", new OperatorExpr(new Operator(ADDITION),
                                              {new Variable("x2ym"),
                                               new OperatorExpr(new Operator(MULTIPLICATION),
                                                                {new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new Variable("i")),
                                                                 new MatrixElementRef(
                                                                     new Variable("x"),
                                                                     new LiteralInt(0),
                                                                     new Variable("i")),
                                                                 new MatrixElementRef(
                                                                     new Variable("y"),
                                                                     new LiteralInt(0),
                                                                     new Variable("i"))}),
                                               new OperatorExpr(new Operator(MULTIPLICATION),
                                                                {new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new OperatorExpr(new Operator(ADDITION),
                                                                                     {new Variable("i"),
                                                                                      new LiteralInt(1)})),
                                                                 new MatrixElementRef(
                                                                     new Variable("x"),
                                                                     new LiteralInt(0),
                                                                     new OperatorExpr(new Operator(ADDITION),
                                                                                      {new Variable("i"),
                                                                                       new LiteralInt(1)})),
                                                                 new MatrixElementRef(
                                                                     new Variable("y"),
                                                                     new LiteralInt(0),
                                                                     new OperatorExpr(new Operator(ADDITION),
                                                                                      {new Variable("i"),
                                                                                       new LiteralInt(1)}))}),
                                               new OperatorExpr(new Operator(MULTIPLICATION),
                                                                {new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new OperatorExpr(new Operator(ADDITION),
                                                                                     {new Variable("i"),
                                                                                      new LiteralInt(2)})),
                                                                 new MatrixElementRef(
                                                                     new Variable("x"),
                                                                     new LiteralInt(0),
                                                                     new OperatorExpr(new Operator(ADDITION),
                                                                                      {new Variable("i"),
                                                                                       new LiteralInt(2)})),
                                                                 new MatrixElementRef(
                                                                     new Variable("y"),
                                                                     new LiteralInt(0),
                                                                     new OperatorExpr(new Operator(ADDITION),
                                                                                      {new Variable("i"),
                                                                                       new LiteralInt(2)}))})
                                              })));

  unwindedForLoopBlock->addChild(new VarAssignm("i",
                                                new OperatorExpr(new Operator(ADDITION),
                                                                 {new Variable("i"), new LiteralInt(3)})));

  unrolledLoopsBlock->addChild(new For(nullptr,
                                       new OperatorExpr(new Operator(LOGICAL_AND), {
                                           new OperatorExpr(new Operator(SMALLER),
                                                            {new Variable("i"),
                                                             new GetMatrixSize(new Variable("x"), new LiteralInt(1))}),
                                           new OperatorExpr(new Operator(SMALLER),
                                                            {new OperatorExpr(new Operator(ADDITION),
                                                                              {new Variable("i"),
                                                                               new LiteralInt(1)}),
                                                             new GetMatrixSize(new Variable("x"), new LiteralInt(1))}),
                                           new OperatorExpr(new Operator(SMALLER),
                                                            {new OperatorExpr(new Operator(ADDITION),
                                                                              {new Variable("i"),
                                                                               new LiteralInt(2)}),
                                                             new GetMatrixSize(new Variable("x"), new LiteralInt(1))})
                                       }),
                                       nullptr,
                                       unwindedForLoopBlock));

  // CLEANUP LOOP
  auto forLoopBlock = new Block();
  forLoopBlock->addChild(
      new VarAssignm("xym", new OperatorExpr(new Operator(ADDITION),
                                             {new Variable("xym"),
                                              new OperatorExpr(new Operator(MULTIPLICATION),
                                                               {new MatrixElementRef(
                                                                   new Variable("x"),
                                                                   new LiteralInt(0),
                                                                   new Variable("i")),
                                                                new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new Variable("i"))})})));
  forLoopBlock->addChild(
      new VarAssignm("x2ym", new OperatorExpr(new Operator(ADDITION),
                                              {new Variable("x2ym"),
                                               new OperatorExpr(new Operator(MULTIPLICATION),
                                                                {new MatrixElementRef(
                                                                    new Variable("x"),
                                                                    new LiteralInt(0),
                                                                    new Variable("i")),
                                                                 new MatrixElementRef(
                                                                     new Variable("x"),
                                                                     new LiteralInt(0),
                                                                     new Variable("i")),
                                                                 new MatrixElementRef(
                                                                     new Variable("y"),
                                                                     new LiteralInt(0),
                                                                     new Variable("i"))})})));
  unrolledLoopsBlock->addChild(new For(nullptr,
                                       new OperatorExpr(new Operator(SMALLER),
                                                        {new Variable("i"),
                                                         new GetMatrixSize(new Variable("x"), new LiteralInt(1))}),
                                       new VarAssignm("i", new OperatorExpr(new Operator(ADDITION),
                                                                            {new Variable("i"), new LiteralInt(1)})),
                                       forLoopBlock));

  func->addStatement(new Return({new Variable("xym"), new Variable("x2ym")}));

  ast.setRootNode(func);
}

VecInt2D EvaluationAlgorithms::runLaplacianSharpeningAlgorithm(VecInt2D img) {
  VecInt2D weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  VecInt2D img2(img.size(), std::vector<int>(img.size())); //TODO: What is actually CORRECT for laplacian sharpening?
  for (int x = 1; x < img.size() - 1; ++x) {
    for (int y = 1; y < img.at(x).size() - 1; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
//          std::cout << x << ", " << y << ", " << i << ", " << j << std::endl;
          value = value + weightMatrix.at(i + 1).at(j + 1)*img.at(x + i).at(y + j);
        }
      }
      img2[x][y] = img.at(x).at(y) - (value/2); //FIXME: This is an integer divison
    }
  }
  return img2;
}

void EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAst(Ast &ast) {
  // -- source code --
  // /// \param img A quadratic image given as row vector (single row matrix) consisting of concatenated rows.
  // /// \param imgSize The image's size. Assumes that img is quadratic, i.e., img has dimension (imgSize, imgSize).
  // VecInt2D runLaplacianSharpeningAlgorithm(Vector<secret_int> img, int imgSize) {
  //     Vector<int> img2;
  //     Matrix<int> weightMatrix = [1 1 1; 1 -8 1; 1 1 1];
  //     for (int x = 1; x < imgSize - 1; ++x) {
  //         for (int y = 1; y < imgSize - 1; ++y) {
  //             int value = 0;
  //             for (int j = -1; j < 2; ++j) {
  //                 for (int i = -1; i < 2; ++i) {
  //                     value = value + weightMatrix[i+1][j+1] * img[imgSize*(x+i)+y+j];
  //                 }
  //             }
  //             img2[imgSize*x+y] = img[imgSize*x+y] - (value/2);
  //         }
  //     }
  //     return img2;
  // }
  auto func = new Function("runLaplacianSharpeningAlgorithm");
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("img")));
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, false), new Variable("imgSize")));

  // std::vector<int> img2;  // has dimension (1,1024)
  // FIXME:
//  func->addStatement(new VarDecl("img2", new Datatype(Types::INT, true),
//                                 new LiteralInt(
//                                     new Matrix<int>(std::vector<std::vector<int>>(1, std::vector<int>(1024))))));
  func->addStatement(new VarDecl("img2", new Datatype(Types::INT)));

  // Matrix<int> weightMatrix = [1 1 1; 1 -8 1; 1 1 1];  –- row-wise concatenation of the original matrix
  func->addStatement(new VarDecl("weightMatrix", new Datatype(Types::INT),
                                 new LiteralInt(new Matrix<int>({{1, 1, 1},
                                                                 {1, -8, 1},
                                                                 {1, 1, 1}}))));

  // value = value + weightMatrix[i+1][j+1] * img[imgSize*(x+i)+y+j]; -- innermost loop body
  auto wmTerm = new MatrixElementRef(new Variable("weightMatrix"),
                                     new OperatorExpr(new Operator(ADDITION), {new Variable("i"), new
                                         LiteralInt(1)}),
                                     new OperatorExpr(new Operator(ADDITION), {new Variable("j"), new
                                         LiteralInt(1)}));
  auto imgTerm = new MatrixElementRef(new Variable("img"),
                                      new LiteralInt(0),  // as img is a single row vector
                                      new OperatorExpr(
                                          new Operator(ADDITION),
                                          {new OperatorExpr(new Operator(MULTIPLICATION),
                                                            {new Variable("imgSize"),
                                                             new OperatorExpr(
                                                                 new Operator(ADDITION),
                                                                 {new Variable("x"),
                                                                  new Variable("i")})}),
                                           new OperatorExpr(new Operator(ADDITION),
                                                            {new Variable("y"),
                                                             new Variable("j")})}));

  auto fourthLoopBody = new Block(new VarAssignm("value",
                                                 new OperatorExpr(new Operator(ADDITION),
                                                                  {new Variable("value"),
                                                                   new OperatorExpr(new Operator(MULTIPLICATION),
                                                                                    {wmTerm,
                                                                                     imgTerm})})));
  // for (int i = -1; i < 2; ++i)  -- 4th level loop
  auto thirdLoopBody = new Block(new For(new VarDecl("i", -1),
                                         new LogicalExpr(new Variable("i"), SMALLER, new LiteralInt(2)),
                                         new VarAssignm("i",
                                                        new OperatorExpr(new Operator(ADDITION),
                                                                         {new Variable("i"),
                                                                          new LiteralInt(1)})),
                                         fourthLoopBody));

  // includes the 3rd level loop
  auto secondLoopBody = new Block({
                                      // int value = 0;
                                      new VarDecl("value", 0),
                                      // for (int j = -1; j < 2; ++j) {...}  -- 3rd level loop
                                      new For(new VarDecl("j", -1),
                                              new LogicalExpr(new Variable("j"), SMALLER, new LiteralInt(2)),
                                              new VarAssignm("j",
                                                             new OperatorExpr(new Operator(ADDITION),
                                                                              {new Variable("j"),
                                                                               new LiteralInt(1)})),
                                              thirdLoopBody),
                                      // img2[imgSize*x+y] = img[imgSize*x+y] - (value/2);
                                      new MatrixAssignm(new MatrixElementRef(new Variable("img2"),
                                                                             new LiteralInt(0),
                                                                             new OperatorExpr(
                                                                                 new Operator(ADDITION),
                                                                                 {new OperatorExpr(
                                                                                     new Operator(MULTIPLICATION),
                                                                                     {new Variable("imgSize"),
                                                                                      new Variable("x")}),
                                                                                  new Variable("y")})),
                                                        new OperatorExpr(
                                                            new Operator(SUBTRACTION),
                                                            {new MatrixElementRef(new Variable("img"),
                                                                                  new LiteralInt(0),
                                                                                  new OperatorExpr(
                                                                                      new Operator(ADDITION),
                                                                                      {new OperatorExpr(
                                                                                          new Operator(MULTIPLICATION),
                                                                                          {new Variable("imgSize"),
                                                                                           new Variable("x")}),
                                                                                       new Variable("y")})),

                                                             new OperatorExpr(
                                                                 new Operator(DIVISION),
                                                                 {new Variable("value"),
                                                                  new LiteralInt(2)})}))});

  // for (int y = 1; y < imgSize - 1; ++y)  -- 2nd level loop
  auto firstLoopBody = new Block(new For(new VarDecl("y", 1),
                                         new LogicalExpr(new Variable("y"),
                                                         SMALLER,
                                                         new OperatorExpr(new Operator(SUBTRACTION),
                                                                          {new Variable("imgSize"),
                                                                           new LiteralInt(1)})),
                                         new VarAssignm("y",
                                                        new OperatorExpr(new Operator(ADDITION),
                                                                         {new Variable("y"),
                                                                          new LiteralInt(1)})),
                                         secondLoopBody));

  // for (int x = 1; x < imgSize - 1; ++x)  -- 1st level loop
  func->addStatement(new For(new VarDecl("x", 1),
                             new LogicalExpr(new Variable("x"),
                                             SMALLER,
                                             new OperatorExpr(new Operator(SUBTRACTION),
                                                              {new Variable("imgSize"), new LiteralInt(1)})),
                             new VarAssignm("x", new OperatorExpr(new Operator(ADDITION),
                                                                  {new Variable("x"), new LiteralInt(1)})),
                             firstLoopBody));

  // return img2;
  func->addStatement(new Return(new Variable("img2")));

  ast.setRootNode(func);

}

void EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAstAfterCtes(Ast &ast) {
  // [BFV-compatible variant without division]
  // VecInt2D runLaplacianSharpeningAlgorithm(Vector<secret_int> img, int imgSize) {
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

  func->addStatement(new VarDecl("img2", new Datatype(Types::INT, true)));

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

  ast.setRootNode(func);
}

std::vector<int> EvaluationAlgorithms::runSobelFilter(const std::vector<int> &img) {
  // ATTENTION: This algorithm works but has not been tested with real values yet!

  // a 3rd-degree polynomial approximation of the square root given as:
  //   sqrt(x) = x * 2.214 + x^2 * -1.098 + x^3 * 0.173
  auto sqrt = [](double x) -> double {
    return x*2.214 + x*x*(-1.098) + x*x*x*0.173;
  };
  VecInt2D F{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  std::vector<int> h(img.size(), 0), v(img.size(), 0), Ix, Iy;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // rot = image << (i*64+j)
      // create a copy of img as rotation using std::rotate always happens in-place
      auto rot = std::vector<int>(img);
      cyclicRotate(rot, i*img.size() + j);

      // h = rot * constant(scale, F[i][j])
      for (auto k = 0; k < rot.size(); ++k) h[k] = rot[k]*F[i][j];

      // v = rot * constant(scale, F[j][i])
      for (auto k = 0; k < rot.size(); ++k) v[k] = rot[k]*F[j][i];

      if (i==0 && j==0) {
        Ix = h;
        Iy = v;
      } else {
        // Ix = Ix + h
        std::transform(Ix.begin(), Ix.end(), h.begin(), Ix.begin(), std::plus<>());
        // Iy = Iy + v
        std::transform(Iy.begin(), Iy.end(), v.begin(), Iy.begin(), std::plus<>());
      }
    }
  }
  // Ix = Ix*Ix
  std::transform(Ix.begin(), Ix.end(), Ix.begin(), Ix.begin(), std::multiplies<>());
  // Iy = Iy*Iy
  std::transform(Iy.begin(), Iy.end(), Iy.begin(), Iy.begin(), std::multiplies<>());

  // result = Ix + Iy
  std::vector<int> result;
  result.reserve(Ix.size());
  std::transform(Ix.begin(), Ix.end(), Iy.begin(), std::back_inserter(result), std::plus<>());

  // result = sqrt(result[0]) + sqrt(result[1]) + ... + sqrt(result[N])
  std::transform(result.begin(), result.end(), result.begin(), [&sqrt](int elem) {
    return sqrt(elem);
  });

  return result;
}

void EvaluationAlgorithms::genSobelFilterAst(Ast &ast) {
  // computeSobelFilter(Vector<int> img) {
  //    Vector<int> F = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  //    Vector<int> h(img.size(), 0);
  //    Vector<int> v(img.size(), 0);
  //    Vector<int> Ix;
  //    Vector<int> Iy;
  //    for (int i = 0; i < 3; ++i) {
  //       for (int j = 0; j < 3; ++j) {
  //          Vector<int> rot = cyclicRotate(img, i*img.size() + j);
  //          for (int k = 0; k < rot.size(); k = k +1) {
  //             h[k] = rot[k]*F[3*i+j];
  //             v[k] = rot[k]*F[3*j+i];
  //          }
  //          if (i==0 && j==0) {
  //             Ix = h;
  //             Iy = v;
  //          } else {
  //             for (int k = 0; k < img.size(); k = k+1) {
  //                Ix[k] = Ix[k] + h[k];
  //                Iy[k] = Iy[k] + v[k];
  //             }
  //          }
  //       }
  //    }
  //    Vector<int> result;
  //    for (int m = 0; m < img.size(); m = m+1) {
  //       result[m] = Ix[m]*Ix[m] + Iy[m]+Iy[m];
  //    }
  //    return result;
  // }
  //
  // Client then computes sqrt(result) to obtain image.

  auto func = new Function("computeSobelFilter");
  func->addParameter(new FunctionParameter(new Datatype(Types::INT, true), new Variable("img")));

  func->addStatement(new VarDecl("F", new Datatype(Types::INT),
                                 new LiteralInt(new Matrix<int>({{-1, 0, 1, -2, 0, 2, -1, 0, 1}}))));
  func->addStatement(new VarDecl("h", new Datatype(Types::INT)));
  func->addStatement(new VarDecl("v", new Datatype(Types::INT)));
  func->addStatement(new VarDecl("Ix", new Datatype(Types::INT)));
  func->addStatement(new VarDecl("Iy", new Datatype(Types::INT)));

  auto outerForLoopBlock = new Block();
  func->addStatement(new For(
      new VarDecl("i", new Datatype(Types::INT), new LiteralInt(0)),
      new OperatorExpr(new Operator(SMALLER), {new Variable("i"), new LiteralInt(3)}),
      new VarAssignm("i", new OperatorExpr(new Operator(ADDITION), {new Variable("i"), new LiteralInt(1)})),
      outerForLoopBlock));

  auto innerForLoopBlock = new Block();
  innerForLoopBlock->addChild(
      new VarDecl("rot", new Datatype(Types::INT), new Rotate(new Variable("img"),
                                                              new OperatorExpr(
                                                                  new Operator(ADDITION),
                                                                  {new OperatorExpr(new Operator(MULTIPLICATION),
                                                                                    {new Variable("i"),
                                                                                     new GetMatrixSize(
                                                                                         new Variable("img"),
                                                                                         new LiteralInt(1))}),
                                                                   new Variable("j")}))));

  auto hvForLoopBlock = new Block();
  hvForLoopBlock->addChild(new MatrixAssignm(
      new MatrixElementRef(new Variable("h"), new LiteralInt(0), new Variable("k")),
      new OperatorExpr(new Operator(MULTIPLICATION),
                       {new MatrixElementRef(new Variable("rot"), new LiteralInt(0), new Variable("k")),
                        new MatrixElementRef(new Variable("F"), new LiteralInt(0),
                                             new OperatorExpr(new Operator(ADDITION),
                                                              {new OperatorExpr(new Operator(MULTIPLICATION),
                                                                                {new LiteralInt(3), new Variable("i")}),
                                                               new Variable("j")}))})));
  hvForLoopBlock->addChild(new MatrixAssignm(
      new MatrixElementRef(new Variable("v"), new LiteralInt(0), new Variable("k")),
      new OperatorExpr(new Operator(MULTIPLICATION),
                       {new MatrixElementRef(new Variable("rot"), new LiteralInt(0), new Variable("k")),
                        new MatrixElementRef(new Variable("F"), new LiteralInt(0),
                                             new OperatorExpr(new Operator(ADDITION),
                                                              {new OperatorExpr(new Operator(MULTIPLICATION),
                                                                                {new LiteralInt(3), new Variable("j")}),
                                                               new Variable("i")}))})));
  innerForLoopBlock->addChild(
      new For(new VarDecl("k", new Datatype(Types::INT), new LiteralInt(0)),
              new OperatorExpr(new Operator(SMALLER), {new Variable("k"),
                                                       new GetMatrixSize(new Variable("rot"), new LiteralInt(1))}),
              new VarAssignm("k", new OperatorExpr(new Operator(ADDITION), {new Variable("k"), new LiteralInt(1)})),
              hvForLoopBlock));

  auto innerMostForLoopBlock = new Block();
  innerMostForLoopBlock->addChild(new MatrixAssignm(
      new MatrixElementRef(new Variable("Ix"), new LiteralInt(0), new Variable("k")),
      new OperatorExpr(new Operator(ADDITION), {
          new MatrixElementRef(new Variable("Ix"), new LiteralInt(0), new Variable("k")),
          new MatrixElementRef(new Variable("h"), new LiteralInt(0), new Variable("k"))})));
  innerMostForLoopBlock->addChild(new MatrixAssignm(
      new MatrixElementRef(new Variable("Iy"), new LiteralInt(0), new Variable("k")),
      new OperatorExpr(new Operator(ADDITION), {
          new MatrixElementRef(new Variable("Iy"), new LiteralInt(0), new Variable("k")),
          new MatrixElementRef(new Variable("v"), new LiteralInt(0), new Variable("k"))})));

  auto elseBlock = new Block();
  elseBlock->addChild(new For(new VarDecl("k", new Datatype(Types::INT), new LiteralInt(0)),
                              new OperatorExpr(new Operator(SMALLER),
                                               {new Variable("k"),
                                                new GetMatrixSize(new Variable("img"), new LiteralInt(1))}),
                              new VarAssignm("k", new OperatorExpr(new Operator(ADDITION),
                                                                   {new Variable("k"),
                                                                    new LiteralInt(1)})),
                              innerMostForLoopBlock));

  innerForLoopBlock->addChild(new If(new OperatorExpr(new Operator(LOGICAL_AND),
                                                      {new OperatorExpr(new Operator(EQUAL),
                                                                        {new Variable("i"), new LiteralInt(0)}),
                                                       new OperatorExpr(new Operator(EQUAL),
                                                                        {new Variable("j"), new LiteralInt(0)})}),
                                     new Block({new VarAssignm("Ix", new Variable("h")),
                                                new VarAssignm("Iy", new Variable("v"))}),
                                     elseBlock));

  outerForLoopBlock->addChild(new For(
      new VarDecl("j", new Datatype(Types::INT), new LiteralInt(0)),
      new OperatorExpr(new Operator(SMALLER), {new Variable("j"), new LiteralInt(3)}),
      new VarAssignm("j", new OperatorExpr(new Operator(ADDITION), {new Variable("j"), new LiteralInt(1)})),
      innerForLoopBlock));

  func->addStatement(new VarDecl("result", new Datatype(Types::INT)));

  auto sumForLoopBlock = new Block();
  sumForLoopBlock->addChild(
      new MatrixAssignm(new MatrixElementRef(new Variable("result"), new LiteralInt(0), new Variable("m")),
                        new OperatorExpr(new Operator(ADDITION),
                                         {new OperatorExpr(new Operator(MULTIPLICATION),
                                                           {new MatrixElementRef(new Variable("Ix"),
                                                                                 new LiteralInt(0),
                                                                                 new Variable("m")),
                                                            new MatrixElementRef(new Variable("Ix"),
                                                                                 new LiteralInt(0),
                                                                                 new Variable("m"))}),
                                          new OperatorExpr(new Operator(MULTIPLICATION),
                                                           {new MatrixElementRef(new Variable("Iy"),
                                                                                 new LiteralInt(0),
                                                                                 new Variable("m")),
                                                            new MatrixElementRef(new Variable("Iy"),
                                                                                 new LiteralInt(0),
                                                                                 new Variable("m"))})})));
  func->addStatement(new For(
      new VarDecl("m", new Datatype(Types::INT), new LiteralInt(0)),
      new OperatorExpr(new Operator(SMALLER),
                       {new Variable("m"), new GetMatrixSize(new Variable("img"), new LiteralInt(1))}),
      new VarAssignm("m", new OperatorExpr(new Operator(ADDITION), {new Variable("m"), new LiteralInt(1)})),
      sumForLoopBlock));

  func->addStatement(new Return(new Variable("result")));

  ast.setRootNode(func);
}

void EvaluationAlgorithms::genSobelFilterAstAfterCtes(Ast &ast) {
  //  computeSobelFilter(Vector<int> img) {
  //    Vector<int> F = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  //    Vector<int> h(img.size(), 0);
  //    Vector<int> v(img.size(), 0);
  //    Vector<int> Ix;
  //    Vector<int> Iy;
  //    for (int i = 0; i < 3; ++i) {
  //      for (int j = 0; j < 3; ++j) {
  //        Vector<int> rot = cyclicRotate(img, i * img.size() + j);
  //        {
  //          int k = 0;
  //          // partially unrolled loop
  //          for (; k < rot.size() && k + 1 < rot.size() && k + 2 < rot.size();) {
  //            h[k] = rot[k] * F[3 * i + j];
  //            v[k] = rot[k] * F[3 * j + i];
  //            h[k+1] = rot[k+1] * F[3*i+j];
  //            v[k+1] = rot[k+1] * F[3*j+i];
  //            h[k+2] = rot[k+2] * F[3*i+j];
  //            v[k+2] = rot[k+2] * F[3*j+i];
  //            k = k + 3;
  //          }
  //          // cleanup loop
  //          for (; k < rot.size(); k = k + 1) {
  //            h[k] = rot[k] * F[3*i+j];
  //            v[k] = rot[k] * F[3*j+i];
  //          }
  //        }
  //        if (i == 0 && j == 0) {
  //          Ix = h;
  //          Iy = v;
  //        } else {
  //          {
  //            int k = 0;
  //            // partially unrolled loop
  //            for (; k < img.size() && k + 1 < img.size() && k + 2 < img.size();) {
  //              Ix[k] = Ix[k] + h[k];
  //              Iy[k] = Iy[k] + v[k];
  //              Ix[k+1] = Ix[k+1] + h[k+1];
  //              Iy[k+1] = Iy[k+1] + v[k+1];
  //              Ix[k+2] = Ix[k+2] + h[k+2];
  //              Iy[k+2] = Iy[k+2] + v[k+2];
  //              k = k + 3;
  //            }
  //            // cleanup loop
  //            for (; k < img.size(); k = k + 1) {
  //              Ix[k] = Ix[k] + h[k];
  //              Iy[k] = Iy[k] + v[k];
  //            }
  //          }
  //        }
  //      }
  //    }
  //
  //    Vector<int> result;
  //    for (int m = 0; m < img.size(); m = m + 1) {
  //      result[m] = Ix[m] * Ix[m] + Iy[m] + Iy[m];
  //    }
  //
  //    return result;
  //  }
  //
  //  Client then computes sqrt(result) to obtain image.
}

#ifdef HAVE_SEAL_BFV
/// SecretKey() actually works, ptr for consistency
std::unique_ptr<seal::SecretKey> secretKey = nullptr;

/// The default constructor used by SEAL in PublicKey() segfaults. Therefore, it's a ptr
std::unique_ptr<seal::PublicKey> publicKey = nullptr;

/// The default constructor used by SEAL in GaloisKey() segfaults. Therefore, it's a ptr
std::unique_ptr<seal::GaloisKeys> galoisKeys = nullptr;

/// the seal context, i.e. object that holds params/etc
std::shared_ptr<seal::SEALContext> context;

void EvaluationAlgorithms::encryptedLaplacianSharpeningAlgorithmBatched(VecInt2D img) {
//  // time measurements
//  std::chrono::microseconds tTotal;
//  auto tStart = std::chrono::high_resolution_clock::now();

  setup_context(context, secretKey, publicKey, galoisKeys);
  auto encoder = seal::BatchEncoder(context);
  auto encryptor = seal::Encryptor(context, *publicKey, *secretKey); //secret Key encryptor is more efficient

  // Encrypt input
  std::vector<int64_t> img_as_vec;
  img_as_vec.reserve(img.size()*img.size());
  for (int x = 1; x < img.size() - 1; ++x) {
    for (int y = 1; y < img.at(x).size() - 1; ++y) {
      img_as_vec.push_back(img[x][y]);
    }
  }
  seal::Plaintext img_ptxt;
  encoder.encode(img_as_vec, img_ptxt);
  seal::Ciphertext img_ctxt(context);
  encryptor.encrypt_symmetric(img_ptxt, img_ctxt);


  // Compute sharpening filter
  auto evaluator = seal::Evaluator(context);

  // Take copies of the image, rotate and mult with weights
  //TODO: Not 100% sure if weight and rotation order match up correctly. But doesn't matter for benchmarking
  std::vector<int> weights = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  std::vector<size_t> rotations =
      {0, 1, 2, img.size(), img.size() + 1, img.size() + 2, 2*img.size(), 2*img.size() + 1, 2*img.size() + 2};
  seal::Plaintext w_ptxt;
  std::vector<seal::Ciphertext> img_ctxts(img.size(), seal::Ciphertext(context));
  img_ctxts.reserve(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    img_ctxts[i] = (i==0) ? std::move(img_ctxt) : img_ctxts[0]; //move for i == 0 saves one ctxt copy
    evaluator.rotate_rows_inplace(img_ctxts[i], (int) rotations[i], *galoisKeys);
//    seal::Ciphertext dst(context);
//    evaluator.rotate_vector(img_ctxts[i], (int) rotations[i], *galoisKeys, dst);
    encoder.encode(std::vector<int64_t>(img_as_vec.size(), weights[i]), w_ptxt);
    evaluator.multiply_plain_inplace(img_ctxts[i], w_ptxt);
  }

  // Sum up all the ctxts
  seal::Ciphertext res_ctxt(context);
  evaluator.add_many(img_ctxts, res_ctxt);

//  auto tEnd = std::chrono::high_resolution_clock::now();
//  tTotal = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
//  std::cout << "Total: " << tTotal.count() << std::endl;
}

void EvaluationAlgorithms::encryptedLaplacianSharpeningAlgorithmNaive(VecInt2D img) {
  // time measurements
//  std::chrono::microseconds tTotal;
//  auto tStart = std::chrono::high_resolution_clock::now();

  setup_context(context, secretKey, publicKey, galoisKeys);
  auto encoder = seal::BatchEncoder(context);
  auto encryptor = seal::Encryptor(context, *publicKey, *secretKey); //secret Key encryptor is more efficient



  // Encrypt input
  std::vector<int64_t> img_as_vec;
  img_as_vec.reserve(img.size()*img.size());
  for (int x = 1; x < img.size() - 1; ++x) {
    for (int y = 1; y < img.at(x).size() - 1; ++y) {
      img_as_vec.push_back(img[x][y]);
    }
  }
  seal::Plaintext img_ptxt;
  encoder.encode(img_as_vec, img_ptxt);
  seal::Ciphertext img_ctxt(context);
  encryptor.encrypt_symmetric(img_ptxt, img_ctxt);

  // Compute sharpening filter
  auto evaluator = seal::Evaluator(context);

  // Naive way: very similar to plain C++
  VecInt2D weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  // Can be default constructed because this is overriden in each loop
  seal::Ciphertext img2_ctxt;

  for (int x = 1; x < img.size() - 1; ++x) {
    for (int y = 1; y < img.at(x).size() - 1; ++y) {
      seal::Ciphertext value(context);
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
//          std::cout << x << ", " << y << ", " << i << ", " << j << std::endl;
          seal::Plaintext w;
          encoder.encode(std::vector<int64_t>(1, weightMatrix.at(i + 1).at(j + 1)), w);
          seal::Ciphertext temp;
          evaluator.rotate_rows(img_ctxt, (x + i)*img.size() + (y + j), *galoisKeys, temp);
          evaluator.multiply_plain_inplace(temp, w);
          evaluator.add_inplace(value, temp);
        }
      }

      seal::Plaintext two;
      encoder.encode(std::vector<int64_t>(1, 2), two);
      seal::Ciphertext temp = img_ctxt;
      evaluator.multiply_plain_inplace(temp, two);
      evaluator.sub_inplace(temp, value);
      //TODO: Add masking and merge masking and mult by two
      //std::vector<int64_t> mask(16384,0);
      //mask[x*img.size()+y] = 1;
      //seal::Plaintext mask_ptxt;
      //encoder.encode(mask,mask_ptxt);
      //evaluator.multiply_plain_inplace(temp,mask_ptxt);
      img2_ctxt = temp;
    }
  }

  auto decryptor = seal::Decryptor(context, *secretKey);
  seal::Plaintext ptxt;
  std::vector<int64_t> output;
  decryptor.decrypt(img2_ctxt, ptxt);
  encoder.decode(ptxt, output);
  for (int k = 1; k < img.size() - 1; ++k) {
    for (int h = 1; h < img.size() - 1; ++h) {
//      std::cout << output[0] << " ";
    }
//    std::cout << std::endl;
  }

//  auto tEnd = std::chrono::high_resolution_clock::now();
//  tTotal = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
//  std::cout << "Total: " << tTotal.count() << std::endl;
}
std::vector<double> EvaluationAlgorithms::runLaplacianSharpeningFilterModified(Matrix<int> &img, int imgSize) {
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
#endif // HAVE_SEAL_BFV

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