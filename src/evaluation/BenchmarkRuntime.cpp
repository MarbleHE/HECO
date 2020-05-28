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
  // │ LAPLACIAN SHARPENING ALGORITHM                                                                                  │
  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

//  std::vector<int> imgSizes = {128};
//  std::vector<int> imgSizes = {8, 16, 32, 64, 96, 128};
//  int numRuns = 5;
//
//
//  std::ofstream resultFile("/Users/patrick/git/master_thesis_report/hpi-thesis/aux/thesis-plots/run-time-benchmarks"
//                           "/Laplacian_OriginalAST_SEAL-OFF-EvalVisitor.csv");
//  if (!resultFile.is_open()) throw std::runtime_error("Could not open file!");
//  resultFile << "#SEAL,CTXT_SLOTS,imgSize,runtimeAvgUs" << std::endl;
//
//  // TODO Try numbers if EvaluationVisitor is used instead...
//
//  for (auto size : imgSizes) {
//    std::chrono::microseconds totalTime{};
//    for (int i = 0; i < numRuns; i = i + 1) {
//      EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAst(ast);
//      std::vector<int> vec(size*size);
//      std::iota(vec.begin(), vec.end(), 0);
//      auto t_start = std::chrono::high_resolution_clock::now();
//      EvaluationVisitor ev({{"img", new LiteralInt(new Matrix<int>({vec}))},
//                            {"imgSize", new LiteralInt(size)}});
//      ev.visit(ast);
////      RuntimeVisitor rtv({{"img", new LiteralInt(new Matrix<int>({vec}))},
////                          {"imgSize", new LiteralInt(size)}});
////      rtv.visit(ast);
//      auto t_end = std::chrono::high_resolution_clock::now();
//      auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
//      totalTime += duration_us;
//      std::cout << duration_us.count() << " μs" << std::endl;
//    }
//    bool seal_found = false;
//#ifdef HAVE_SEAL_BFV
//    seal_found = true;
//#endif
//    resultFile << seal_found << "," << Ciphertext::DEFAULT_NUM_SLOTS
//               << "," << size << "," << totalTime.count()/numRuns << std::endl;
//  }
//  resultFile.close();


//  std::ofstream resultFile2("/Users/patrick/git/master_thesis_report/hpi-thesis/aux/thesis-plots/run-time-benchmarks"
//                           "/Laplacian_CtesdAST_SEAL-OFF.csv");
//  if (!resultFile2.is_open()) throw std::runtime_error("Could not open file!");
//  resultFile2 << "#SEAL,CTXT_SLOTS,imgSize,runtimeAvgUs" << std::endl;
//
//  for (auto size : imgSizes) {
//    std::chrono::microseconds totalTime;
//    for (int i = 0; i < numRuns; i = i + 1) {
//      EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAstAfterCtes(ast);
//      std::vector<int> vec(size*size);
//      std::iota(vec.begin(), vec.end(), 0);
//      auto t_start = std::chrono::high_resolution_clock::now();
//      RuntimeVisitor rtv({{"img", new LiteralInt(new Matrix<int>({vec}))},
//                          {"imgSize", new LiteralInt(size)}});
//      rtv.visit(ast);
//      auto t_end = std::chrono::high_resolution_clock::now();
//      auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
//      totalTime += duration_us;
//      std::cout << duration_us.count() << " μs" << std::endl;
//    }
//    resultFile2 << SEAL_FOUND << "," << Ciphertext::DEFAULT_NUM_SLOTS
//               << "," << size << "," << totalTime.count()/numRuns << std::endl;
//  }
//  resultFile2.close();

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
  // │  SEAL-NATIVE IMPLEMENTATIONS
  //  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  /// Image size
  size_t imgSize = 128;
  std::cout << "EVALUATING Laplacian Sharpening on an Image of size " << imgSize << "x" << imgSize <<"." << std::endl;

 // OPTIMIZED
  std::cout << "OPTIMIZED (RuntimeVisitor):" << std::endl;
  EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAstAfterCtes(ast);



  // a img_size x img_size image encoded as single img_size^2 elements row vector
  auto imgData = genRandomImageData(imgSize, Ciphertext::DEFAULT_NUM_SLOTS);

  // execute the plaintext algorithm to know the expected result
  auto expectedResult = EvaluationAlgorithms::runLaplacianSharpeningFilterModified(*imgData, imgSize);
//  Ciphertext ct = Ciphertext(expectedResult);

  // perform the actual execution by running the RuntimeVisitor
  RuntimeVisitor rt({{"img", new LiteralInt(imgData)}, {"imgSize", new LiteralInt(imgSize)}});
  std::chrono::microseconds totalTime;
  auto t_start = std::chrono::high_resolution_clock::now();
  rt.visit(ast);
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
  std::cout << duration_us.count() << " μs" << std::endl;

#ifdef HAVE_SEAL_BFV
  std::cout << "OPTIMIZED (native SEAL):" << std::endl;
  std::vector<int> vec(imgSize);
  std::iota(vec.begin(), vec.end(), 0);
  std::vector<std::vector<int>> img(imgSize, vec);
  t_start = std::chrono::high_resolution_clock::now();
  EvaluationAlgorithms::encryptedLaplacianSharpeningAlgorithmBatched(img);
  //EvaluationAlgorithms::encryptedLaplacianSharpeningAlgorithmBatched(img));
  t_end = std::chrono::high_resolution_clock::now();
  duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
  std::cout << duration_us.count() << " μs" << std::endl;
#endif


  // UNOPTIMIZED
  std::cout << "UNOPTIMIZED (RuntimeVisitor):" << std::endl;
  EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAst(ast);

  // perform the actual execution by running the RuntimeVisitor
  RuntimeVisitor rt_original({{"img", new LiteralInt(imgData)}, {"imgSize", new LiteralInt(imgSize)}});
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
