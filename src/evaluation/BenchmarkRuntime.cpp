#include <ast_opt/ast/Ast.h>
#include <ast_opt/evaluation/EvaluationAlgorithms.h>
#include <ast_opt/visitor/PrintVisitor.h>
#include <ast_opt/visitor/CompileTimeExpressionSimplifier.h>
#include <ast_opt/visitor/RuntimeVisitor.h>
#include <fstream>
#include <chrono>
#include "ast_opt/evaluation/BenchmarkRuntime.h"
#include "../../test/AstTestingGenerator.h"

#define TRUE "1"
#define FALSE "0"

int main() {
  Ast ast;

  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  // │  LINEAR REGRESSION                                                                                              │
  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  // TODO: Adapt RuntimeVisitor do to FHE computation if VarAssignm (instead of MatrixAssignm) is involved
//  EvaluationAlgorithms::genLinearRegressionAst(ast);
//  EvaluationAlgorithms::genLinearRegressionAstAfterCtes(ast);
//  auto t_start = std::chrono::high_resolution_clock::now();
//  std::vector<int> vec(65536);
//  std::iota(vec.begin(), vec.end(), 0);
//  std::vector<int> vec2(vec.begin(), vec.end());
//  std::reverse(vec2.begin(), vec2.end());
//  RuntimeVisitor rtv({{"dataX",new LiteralInt(new Matrix<int>({vec}))},
//                      {"dataY",new LiteralInt(new Matrix<int>({vec2}))}});
//  rtv.visit(ast);
//  auto t_end = std::chrono::high_resolution_clock::now();
//  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
//  std::cout << duration_us.count() << " μs" << std::endl;

  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  // │  POLYNOMIAL REGRESSION                                                                                          │
  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

//  EvaluationAlgorithms::genPolynomialRegressionAstAfterCtes(ast);
//
////  CompileTimeExpressionSimplifier ctes;
////  ctes.visit(ast);
//
//  PrintVisitor pv;
//  pv.visit(ast);
//
////  EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAstAfterCtes(ast);
//  auto t_start = std::chrono::high_resolution_clock::now();
//  std::vector<int> vec(65536);
//  std::iota(vec.begin(), vec.end(), 0);
//  std::vector<int> vec2(vec.begin(), vec.end());
//  std::reverse(vec2.begin(), vec2.end());
//  RuntimeVisitor rtv({{"x",new LiteralInt(new Matrix<int>({vec}))},
//                      {"y",new LiteralInt(new Matrix<int>({vec2}))}});
//  rtv.visit(ast);
//  auto t_end = std::chrono::high_resolution_clock::now();
//  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
//  std::cout << duration_us.count() << " μs" << std::endl;

  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  // │  SOBEL FILTER                                                                                                   │
  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

//  EvaluationAlgorithms::genSobelFilterAst(ast);
//
//  CompileTimeExpressionSimplifier ctes(CtesConfiguration(2));
//  ctes.visit(ast);
//
//  PrintVisitor pv;
//  pv.visit(ast);
//
////  EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAstAfterCtes(ast);
//  auto t_start = std::chrono::high_resolution_clock::now();
//  std::vector<int> vec(65536);
//  std::iota(vec.begin(), vec.end(), 0);
//  std::vector<int> vec2(vec.begin(), vec.end());
//  std::reverse(vec2.begin(), vec2.end());
//  RuntimeVisitor rtv({{"x",new LiteralInt(new Matrix<int>({vec}))},
//                      {"y",new LiteralInt(new Matrix<int>({vec2}))}});
//  rtv.visit(ast);
//  auto t_end = std::chrono::high_resolution_clock::now();
//  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
//  std::cout << duration_us.count() << " μs" << std::endl;

  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  // │ LAPLACIAN SHARPENING ALGORITHM                                                                                  │
  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  /// Image size
  size_t imgSize = 32;

  // Timers
  decltype(std::chrono::high_resolution_clock::now()) t_start;
  decltype(std::chrono::high_resolution_clock::now()) t_end;
  decltype(std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)) duration_us;

  std::cout << "EVALUATING Laplacian Sharpening on an Image of size " << imgSize << "x" << imgSize << "." << std::endl;

  // a img_size x img_size image encoded as single img_size^2 elements row vector
  auto imgData = genRandomImageData(imgSize, Ciphertext::DEFAULT_NUM_SLOTS);
  std::vector<int> vec(imgSize);
  std::iota(vec.begin(), vec.end(), 0);
  std::vector<std::vector<int>> img(imgSize, vec);

  // execute the plaintext algorithm to know the expected result
  auto expectedResult = EvaluationAlgorithms::runLaplacianSharpeningFilterModified(*imgData, imgSize);

  // OPTIMIZED
  std::cout << "OPTIMIZED (RuntimeVisitor):" << std::endl;
  EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAstAfterCtes(ast);
  RuntimeVisitor rt_opt({{"img", new LiteralInt(imgData)}, {"imgSize", new LiteralInt(imgSize)}});
  std::chrono::microseconds totalTime;
  t_start = std::chrono::high_resolution_clock::now();
  rt_opt.visit(ast);
  t_end = std::chrono::high_resolution_clock::now();
  duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
  std::cout << duration_us.count() << " μs" << std::endl;

#ifdef HAVE_SEAL_BFV
  std::cout << "OPTIMIZED (native SEAL):" << std::endl;
  t_start = std::chrono::high_resolution_clock::now();
  EvaluationAlgorithms::encryptedLaplacianSharpeningAlgorithmBatched(img);
  t_end = std::chrono::high_resolution_clock::now();
  duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
  std::cout << duration_us.count() << " μs" << std::endl;
#endif


  // UNOPTIMIZED
  std::cout << "UNOPTIMIZED (RuntimeVisitor):" << std::endl;
  EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAst(ast);
  RuntimeVisitor rt_original({{"img", new LiteralInt(imgData)}, {"imgSize", new LiteralInt(imgSize)}});
  rt_original.disableBatchingOpt = true;
  t_start = std::chrono::high_resolution_clock::now();
  rt_original.visit(ast);
  t_end = std::chrono::high_resolution_clock::now();
  duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
  std::cout << duration_us.count() << " μs" << std::endl;

#ifdef HAVE_SEAL_BFV
  std::cout << "UNOPTIMIZED (native SEAL):" << std::endl;
  t_start = std::chrono::high_resolution_clock::now();
  EvaluationAlgorithms::encryptedLaplacianSharpeningAlgorithmNaive(img);
  t_end = std::chrono::high_resolution_clock::now();
  duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
  std::cout << duration_us.count() << " μs" << std::endl;
#endif
  return 0;
}
