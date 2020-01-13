#ifndef MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_
#define MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_

#include <map>
#include <string>
#include <iostream>
#include "../include/ast/Ast.h"
#include "RandNumGen.h"

static void astOutputComparer(Ast &unmodifiedAst, Ast &rewrittenAst, unsigned int seed, int numTestRuns,
                              std::map<std::string, Literal*> &evalParams) {
  // create random number generator with test-specific seed
  RandLiteralGen rng(seed);

  for (int i = 0; i < numTestRuns; i++) {
    // generate new parameter values
    rng.randomizeValues(evalParams);

    // evaluate both ASTs with previously generated params
    auto resultExpected = unmodifiedAst.evaluate(evalParams, false);
    auto resultRewrittenAst = rewrittenAst.evaluate(evalParams, false);

    // compare results of original and rewritten AST
    ASSERT_EQ(*resultExpected, *resultRewrittenAst);
  }
}

#endif //MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_
