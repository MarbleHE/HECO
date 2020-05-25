#include "ast_opt/evaluation/BenchmarkCtes.h"
#include <ast_opt/visitor/CompileTimeExpressionSimplifier.h>
#include <ast_opt/visitor/PrintVisitor.h>
#include "ast_opt/ast/Ast.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/VarAssignm.h"
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/visitor/CompileTimeExpressionSimplifier.h"
#include <chrono>

Ast BenchmarkCtes::createShallowAstOfKnownValuesOnly(int numAssignments) {
  // int foo(int a) {
  //    int a = 8;
  //    int b = 17;
  //    int result = 1;
  //    result = result * (a + b);  // numAssignments times
  //    result = result * (a + b)
  //    ...
  //    return result;
  // }
  auto func = new Function("foo");
  func->addStatement(new VarDecl("a", new Datatype(Types::INT), new LiteralInt(8)));
  func->addStatement(new VarDecl("b", new Datatype(Types::INT), new LiteralInt(17)));
  func->addStatement(new VarDecl("result", new Datatype(Types::INT), new LiteralInt(1)));
  for (int i = 0; i < numAssignments; ++i) {
    func->addStatement(new VarAssignm("result", new OperatorExpr(new Operator(MULTIPLICATION), {
        new Variable("result"), new OperatorExpr(new Operator(ADDITION), {new Variable("a"), new Variable("b")})
    })));
  }
  func->addStatement(new Return(new Variable("result")));
  return Ast(func);
}

Ast BenchmarkCtes::createDeepAstWithUnknownValues(int N) {
  // foo(int a) {
  //    int sum = 0;
  //    // repeat N times:
  //    sum = sum + (a * 10);
  //    sum = sum * (a + 11);
  //    // end of repeated statements
  //    return sum;
  // }
  auto func = new Function("foo");
  func->addParameter(new FunctionParameter(new Datatype(Types::INT), new Variable("a")));
  func->addStatement(new VarDecl("sum", new Datatype(Types::INT), new LiteralInt(0)));

  for (int i = 0; i < N; ++i) {
    func->addStatement(
        new VarAssignm("sum", new OperatorExpr(new Operator(ADDITION),
                                               {new Variable("sum"),
                                                new OperatorExpr(new Operator(MULTIPLICATION),
                                                                 {new Variable("a"),
                                                                  new LiteralInt(10)})})));
    func->addStatement(new VarAssignm("sum", new OperatorExpr(new Operator(MULTIPLICATION),
                                                              {new Variable("sum"),
                                                               new OperatorExpr(new Operator(ADDITION),
                                                                                {new Variable("a"),
                                                                                 new LiteralInt(11)})})));
  }

  func->addStatement(new VarDecl("a", new Datatype(Types::INT), new LiteralInt(42)));
  func->addStatement(new Return(new Variable("sum")));
  return Ast(func);
}

void BenchmarkCtes::runBenchmarkAndPrintResult(std::function<Ast(int)> &func, const std::vector<int> &sizes, int
numRunsPerSize) {
  std::chrono::microseconds accumulatedTimeUs{0};
  for (int curSize : sizes) {
    for (int j = 0; j < numRunsPerSize; ++j) {
      auto ast = func(curSize);
      auto t_start = std::chrono::high_resolution_clock::now();
      CompileTimeExpressionSimplifier ctes;
      ctes.visit(ast);
      auto t_end = std::chrono::high_resolution_clock::now();
      auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
      accumulatedTimeUs += duration_us;
    }
    accumulatedTimeUs /= numRunsPerSize;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(accumulatedTimeUs);
    std::cout << curSize << "," << accumulatedTimeUs.count() << ","
              << duration_ms.count() << std::endl;
  }
}

int main() {
//  std::vector<int> sizes{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1'024, 2'048, 4'096, 8'192, 16'384, 32'768, 65'536};

  //  ============== AST With Known Values Only ==============

  std::vector<int> sizes{0,
                         5'000,
                         10'000,
                         15'000,
                         20'000,
                         25'000,
                         30'000,
                         35'000,
                         40'000,
                         45'000,
                         50'000,
                         55'000,
                         60'000,
                         65'000,
                         70'000,
                         75'000,
                         80'000,
                         85'000,
                         90'000,
                         95'000,
                         100'000,
  };
  int numRunsPerSize = 5;

//  std::function<Ast(int)> f = &BenchmarkCtes::createShallowAstOfKnownValuesOnly;
//  BenchmarkCtes::runBenchmarkAndPrintResult(f, sizes, numRunsPerSize);

  //  ============== AST With Unknown Values Only ==============

  std::vector<int> sizes2{
//      1,
//      10,
//      25,
//      50,
//      75,
//      100,
//      150,
//      200,
//      500,
//      750,
//                         1'000,
//      1250,
//      1500,
//      1750,
//                         2'000,
//    1250,
//    1500,
      1750,
//    2000,
//    2250
  };
  int numRunsPerSize2 = 2;
  std::function<Ast(int)> f = &BenchmarkCtes::createDeepAstWithUnknownValues;
  BenchmarkCtes::runBenchmarkAndPrintResult(f, sizes2, numRunsPerSize2);

  return 0;
}


