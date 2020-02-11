#ifndef MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_
#define MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_

#include "gtest/gtest.h"
#include <map>
#include <string>
#include <iostream>
#include "../include/ast/Ast.h"
#include "RandNumGen.h"

/// Defines the default number of test runs to be performed, if not passed as parameter to astOutputComparer.
/// Decides whether exhaustive testing (i.e., comparing all possible inputs) will be used or not.
/// 2^12 = 4,096
const static int CIRCUIT_MAX_TEST_RUNS = 4'096;

static void astOutputComparer(Ast &unmodifiedAst, Ast &rewrittenAst, unsigned int seed, int numTestRuns,
                              std::map<std::string, Literal*> &evalParams) {
  // create random number generator with test-specific seed
  RandLiteralGen rng(seed);

  for (int i = 0; i < numTestRuns; i++) {
    // generate new parameter values
    rng.randomizeValues(evalParams);

    // evaluate both ASTs with previously generated params
    auto resultExpected = unmodifiedAst.evaluateAst(evalParams, false);
    auto resultRewrittenAst = rewrittenAst.evaluateAst(evalParams, false);

    // compare results of original and rewritten AST
    ASSERT_EQ(*resultExpected, *resultRewrittenAst);
  }
}

/// This class provides a way for exhaustive testing for circuits with binary inputs.
/// Its implementation is naive and not very efficient. However, exhaustive testing should anyway only be performed if
/// the number of circuit inputs is small.
struct LiteralBoolCombinationsGen {
 public:
  std::vector<LiteralBool*> params;
  std::bitset<CIRCUIT_MAX_TEST_RUNS> nextBitCombination;

  void updateParams() {
    // Update all LiteralBools in params based on the current bit states in nextBitCombination.
    // Iterator 'index' is limited by params.size() because nextBitCombination is larger as its size must be known at
    // compile-time. A better approach would be to initially update all bits but in further iterations remember the last
    // modified bit and based on that exit the loop earlier.
    for (auto index = 0; index < params.size(); ++index) {
      bool bitValue = nextBitCombination[index];
      params.at(index)->setValue(bitValue);
    }
  }

 public:
  explicit LiteralBoolCombinationsGen(std::map<std::string, Literal*> &evalParams) {
    nextBitCombination = std::bitset<CIRCUIT_MAX_TEST_RUNS>(0);
    // convert params into vector of literals
    for (auto &[varIdentifier, literal] : evalParams) params.push_back(literal->castTo<LiteralBool>());
  }

  bool hasNextAndUpdate() {
    // stop if we already built all possible combinations for 2^{params.size()} bits
    if (nextBitCombination.to_ulong() < pow(2, params.size())) {
      // modify params map based on combination in nextBitCombination
      updateParams();

      // increment nextBitCombination map to represent next value
      nextBitCombination = nextBitCombination.to_ulong() + 1l;

      return true;
    }
    return false;
  }
};

static void circuitOutputComparer(Ast &unmodifiedAst, Ast &rewrittenAst, unsigned int seed, int numMaxTestRuns,
                                  std::map<std::string, Literal*> &evalParams) {
  // a function that returns True if the given evalParams entry is a LiteralBool
  auto isLiteralBool = [](const auto &mapEntry) {
    return (dynamic_cast<LiteralBool*>(mapEntry.second) != nullptr);
  };

  // Check if we can perform exhaustive testing. We need to fall-back to randomized testing if:
  // - not all evalParams are LiteralBool (<=> there exists parameter other than type LiteralBool)
  // - or the specified maximum number of test runs would be exceeded, i.e., 2^{|evalParams|} > numMaxTestRuns
  if (!std::all_of(evalParams.begin(), evalParams.end(), isLiteralBool)
      || pow(2, evalParams.size()) > numMaxTestRuns) {
    // in either of theses cases we fall-back to the "traditional" method -> randomized testing
    return astOutputComparer(unmodifiedAst, rewrittenAst, seed, numMaxTestRuns, evalParams);
  } else {
    // we can perform exhaustive testing: create a new LiteralBoolCombinationsGen instance
    LiteralBoolCombinationsGen literalBoolCombinationGen(evalParams);

    // as long as there are combinations of input parameters we have not tested yet -> continue
    while (literalBoolCombinationGen.hasNextAndUpdate()) {

      // perform evaluation of both ASTs using current parameter set
      auto resultExpected = unmodifiedAst.evaluateCircuit(evalParams, false);
      auto resultRewrittenAst = rewrittenAst.evaluateCircuit(evalParams, false);

      // compare results of both ASTs for current param set
      ASSERT_EQ(*resultExpected, *resultRewrittenAst);
    }
  }
}

static void circuitOutputComparer(Ast &unmodifiedAst, Ast &rewrittenAst, unsigned int seed,
                                  std::map<std::string, Literal*> &evalParams) {
  return circuitOutputComparer(unmodifiedAst, rewrittenAst, seed, CIRCUIT_MAX_TEST_RUNS, evalParams);
}

#endif //MASTER_THESIS_CODE_INCLUDE_UTILITIES_TESTUTILS_H_
