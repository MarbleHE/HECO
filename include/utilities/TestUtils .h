#ifndef MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_
#define MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_

#include <map>
#include <string>
#include <iostream>
#include "cmake-build-debug/test/googletest-src/googletest/include/gtest/gtest.h"
#include "../include/ast/Ast.h"
#include "RandNumGen.h"

static void astOutputComparer(Ast &unmodifiedAst, Ast &rewrittenAst, unsigned int seed, int numTestRuns,
                              const std::map<std::string, Literal*> &evalExampleParams) {
  // create random number generator with test-specific seed
  RandLiteralGen rng(seed);

  for (int i = 0; i < numTestRuns; i++) {
    // generate new parameter values
    auto currentParameterSet = rng.getRandomValues(evalExampleParams);

    // evaluate both ASTs with previously generated params
    auto resultExpected = unmodifiedAst.evaluate(currentParameterSet, false);
    auto resultRewrittenAst = rewrittenAst.evaluate(currentParameterSet, false);

    // compare results of original and rewritten AST
    ASSERT_EQ(*resultExpected, *resultRewrittenAst);
  }
}

#endif //MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_
