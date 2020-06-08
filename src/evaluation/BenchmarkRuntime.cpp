#include <ast_opt/ast/Ast.h>
#include <ast_opt/evaluation/EvaluationAlgorithms.h>
#include <ast_opt/visitor/PrintVisitor.h>
#include <ast_opt/visitor/CompileTimeExpressionSimplifier.h>
#include <ast_opt/visitor/RuntimeVisitor.h>
#include <fstream>
#include <chrono>
#include "ast_opt/evaluation/BenchmarkRuntime.h"
#include "../../test/AstTestingGenerator.h"
#include <filesystem>

namespace fs = std::filesystem;

#define TRUE "1"
#define FALSE "0"

int main(int argc, char *argv[]) {

  // read benchmark configuration file
  std::ifstream t(argv[1]);
  std::stringstream buffer;
  buffer << t.rdbuf();

  // parse JSON
  json config = json::parse(buffer.str());

  // extract values
  int numTestruns = config["num_testruns"];
  auto imageSizes = config["image_sizes"];
  auto programs = config["programs"];
  std::string resultDirectory = config["result_files_directory"];

  // execute benchmark
  for (auto const &prog : programs) {
    // create output file
    std::stringstream outputFileName;
    outputFileName << "result_" << prog.get<std::string>() << ".csv";
    fs::path full_path = fs::path(resultDirectory)/fs::path(outputFileName.str());
    std::ofstream resultFile(full_path);
    // write header
    resultFile << "run,imgSize,timeUs" << std::endl;

    for (int i = 1; i <= numTestruns; ++i) {
      for (int imgSize : imageSizes) {
        Ast ast;
        EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAst(ast);

        // create random data as image input
        auto imgData = genRandomImageData(imgSize, Ciphertext::DEFAULT_NUM_SLOTS);
        std::vector<int> vec(imgSize);
        std::iota(vec.begin(), vec.end(), 0);
        std::vector<std::vector<int>> img(imgSize, vec);

        // execute benchmark
        auto t_start = std::chrono::high_resolution_clock::now();
        if (prog=="OPT_RV") {
          RuntimeVisitor rtv({{"img", new LiteralInt(new Matrix<int>({vec}))}, {"imgSize", new LiteralInt(imgSize)}});
          rtv.visit(ast);
        } else if (prog=="OPT_SEAL") {
#ifdef HAVE_SEAL_BFV
          EvaluationAlgorithms::encryptedLaplacianSharpeningAlgorithmBatched(img);
#else
          throw std::runtime_error("Cannot continue as SEAL could not be found!");
#endif
        } else if (prog=="UNOPT_RV") {
          RuntimeVisitor rtv({{"img", new LiteralInt(new Matrix<int>({vec}))}, {"imgSize", new LiteralInt(imgSize)}});
          rtv.disableBatchingOpt = true;
          rtv.visit(ast);
        } else if (prog=="UNOPT_SEAL") {
#ifdef HAVE_SEAL_BFV
          EvaluationAlgorithms::encryptedLaplacianSharpeningAlgorithmNaive(img);
#else
          throw std::runtime_error("Cannot continue as SEAL could not be found!");
#endif
        }
        // stop time
        auto t_end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

        // write result line
        resultFile << i << "," << imgSize << "," << duration_us.count() << std::endl;
      }
    }

    resultFile.close();
  }

  exit(0);

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
  std::vector<int> imgSizes = {8, 16, 32, 64, 96, 128};
  int numRuns = 5;

  std::ofstream resultFile("/Users/patrick/git/master_thesis_report/hpi-thesis/aux/thesis-plots/run-time-benchmarks"
                           "/Laplacian_OriginalAST_SEAL-OFF-EvalVisitor.csv");
  if (!resultFile.is_open()) throw std::runtime_error("Could not open file!");
  resultFile << "#SEAL,CTXT_SLOTS,imgSize,runtimeAvgUs" << std::endl;

  // TODO Try numbers if EvaluationVisitor is used instead...

  for (auto size : imgSizes) {
    std::chrono::microseconds totalTime{};
    for (int i = 0; i < numRuns; i = i + 1) {
      EvaluationAlgorithms::genLaplacianSharpeningAlgorithmAst(ast);
      std::vector<int> vec(size*size);
      std::iota(vec.begin(), vec.end(), 0);
      auto t_start = std::chrono::high_resolution_clock::now();
      EvaluationVisitor ev({{"img", new LiteralInt(new Matrix<int>({vec}))},
                            {"imgSize", new LiteralInt(size)}});
      ev.visit(ast);
//      RuntimeVisitor rtv({{"img", new LiteralInt(new Matrix<int>({vec}))},
//                          {"imgSize", new LiteralInt(size)}});
//      rtv.visit(ast);
      auto t_end = std::chrono::high_resolution_clock::now();
      auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
      totalTime += duration_us;
      std::cout << duration_us.count() << " μs" << std::endl;
    }
//    resultFile << SEAL_FOUND << "," << Ciphertext::DEFAULT_NUM_SLOTS
//               << "," << size << "," << totalTime.count()/numRuns << std::endl;
  }
  resultFile.close();


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
