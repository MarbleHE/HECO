#ifndef AST_OPTIMIZER_SRC_EVALUATION_BENCHMARK_H_
#define AST_OPTIMIZER_SRC_EVALUATION_BENCHMARK_H_

#include <ast_opt/ast/Ast.h>

class BenchmarkCtes {

 public:
  static Ast createDeepAstWithUnknownValues(int N);

  static void runBenchmarkAndPrintResult(std::function<Ast(int)> &func,
                                         const std::vector<int> &sizes,
                                         int numRunsPerSize);

  static Ast createShallowAstOfKnownValuesOnly(int numAssignments);
};

#endif //AST_OPTIMIZER_SRC_EVALUATION_BENCHMARK_H_
