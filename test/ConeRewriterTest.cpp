#include <Ast.h>
#include <ConeRewriter.h>
#include <TestUtils.h>
#include "gtest/gtest.h"
#include "AstTestingGenerator.h"
#include "../include/ast/Variable.h"

class ConeRewriterFixture : public ::testing::Test {
 private:
  std::map<std::string, Literal*>* evaluationParameters;

 protected:
  ConeRewriterFixture() : evaluationParameters(nullptr) {};

  static Ast* generateAst(int demoAstId) {
    Ast* newAst = new Ast;
    AstTestingGenerator::generateAst(19, *newAst);
    return newAst;
  }

  void SetUp() override {}

  virtual void SetUp(std::map<std::string, Literal*> &evalParams) {
    evaluationParameters = &evalParams;
  }

  bool getBoolValue(const std::string &evalParamsKey) {
    try {
      return dynamic_cast<LiteralBool*>(evaluationParameters->at(evalParamsKey))->getValue();
    } catch (std::out_of_range &exception) {
      throw std::out_of_range("");
    }
  }

  bool computeResultForCircuit19() {
    unsigned a11_left = getBoolValue("a_1^(1)_left");
    unsigned a11_right = getBoolValue("a_1^(1)_right");
    unsigned a21_left = getBoolValue("a_2^(1)_left");
    unsigned a21_right = getBoolValue("a_2^(1)_right");
    unsigned a12_left = getBoolValue("a_1^(2)_left");
    unsigned a12_right = getBoolValue("a_1^(2)_right");
    unsigned a22_left = getBoolValue("a_2^(2)_left");
    unsigned a22_right = getBoolValue("a_2^(2)_right");
    unsigned y1 = getBoolValue("y_1");
    unsigned at = getBoolValue("a_t");

    // evaluate expression that represents circuit 19 to manually verify evaluation result
    return (
        ((((a11_left & a11_right) & (a21_left ^ a21_right)) ^ ((a12_left & a12_right) & (a22_left ^ a22_right))) ^ y1)
            & at);
  }

  static void rewriteAndAssertMultiplicativeDepthDecreased(Ast* astToRewrite) {
    // save original multiplicative depth
    MultiplicativeDepthCalculator mdc(*astToRewrite);
    auto originalMaxMultDepth = mdc.getMaximumMultiplicativeDepth();

    // apply cone rewriting to AST
    ConeRewriter coneRewriter(astToRewrite, mdc);
    astToRewrite = &coneRewriter.applyConeRewriting();

    // determine new multiplicative depth
    mdc.precomputeMultDepths(*astToRewrite);
    auto newMaxMultDepth = mdc.getMaximumMultiplicativeDepth();

    // check if multiplicative depth is reduced
    ASSERT_TRUE(originalMaxMultDepth > newMaxMultDepth);
  }

};

TEST_F(ConeRewriterFixture, simpleStaticConeTest) {
  const int DEMO_CIRCUIT_NO = 19;

  // -------–------------------------------------------
  // Check that the multiplicative depth decreased
  // ---------------------------------------------------

  // build demo circuit
  Ast* astToRewrite = generateAst(DEMO_CIRCUIT_NO);

  // do rewriting and check that multiplicative depth decreased
  rewriteAndAssertMultiplicativeDepthDecreased(astToRewrite);

  // -------–------------------------------------------
  // Check that the original circuit and the rewritten are logically equivalent
  // ---------------------------------------------------

  // check if rewritten circuit is logically equivalent to original one
  Ast* originalAst = generateAst(DEMO_CIRCUIT_NO);

  std::map<std::string, Literal*> inputValues =
      {{std::string("a_1^(1)_left"), new LiteralBool(true)},
       {std::string("a_1^(1)_right"), new LiteralBool(true)},
       {std::string("a_2^(1)_left"), new LiteralBool(false)},
       {std::string("a_2^(1)_right"), new LiteralBool(true)},
       {std::string("a_1^(2)_left"), new LiteralBool(true)},
       {std::string("a_1^(2)_right"), new LiteralBool(false)},
       {std::string("a_2^(2)_left"), new LiteralBool(true)},
       {std::string("a_2^(2)_right"), new LiteralBool(true)},
       {std::string("y_1"), new LiteralBool(true)},
       {std::string("a_t"), new LiteralBool(true)}};

  // evaluate both ASTs
  auto resultOriginal = originalAst->evaluateCircuit(inputValues, false);
  auto resultRewritten = astToRewrite->evaluateCircuit(inputValues, false);

  // check evaluation results by verifying that...
  // 1. the circuit in general correctly evaluates by comparison to a trivial implementation
  SetUp(inputValues);
  ASSERT_EQ(*resultOriginal->castTo<LiteralBool>(), LiteralBool(computeResultForCircuit19()));
  // 2. the rewriting does not modify the AST's semantics compared to the original AST (-> logical equivalence)
  ASSERT_EQ(*resultOriginal->castTo<LiteralBool>(), *resultRewritten->castTo<LiteralBool>());
}

TEST_F(ConeRewriterFixture, evaluationBasedTestForSimpleExtendedCircuit) {
  const int DEMO_CIRCUIT_NO = 19;

  // -------–------------------------------------------
  // Check that the multiplicative depth decreased
  // ---------------------------------------------------

  // build demo circuit
  Ast* astToRewrite = generateAst(DEMO_CIRCUIT_NO);

  // do rewriting and check that multiplicative depth decreased
  rewriteAndAssertMultiplicativeDepthDecreased(astToRewrite);

  // -------–------------------------------------------
  // Check that the original circuit and the rewritten are logically equivalent
  // ---------------------------------------------------

  // define expected input parameters and some arbitrary input values
  // (values will be overwritten by circuitOutputComparer)
  std::map<std::string, Literal*> inputValues =
      {{std::string("a_1^(1)_left"), new LiteralBool(true)},
       {std::string("a_1^(1)_right"), new LiteralBool(true)},
       {std::string("a_2^(1)_left"), new LiteralBool(false)},
       {std::string("a_2^(1)_right"), new LiteralBool(true)},
       {std::string("a_1^(2)_left"), new LiteralBool(true)},
       {std::string("a_1^(2)_right"), new LiteralBool(false)},
       {std::string("a_2^(2)_left"), new LiteralBool(true)},
       {std::string("a_2^(2)_right"), new LiteralBool(true)},
       {std::string("y_1"), new LiteralBool(true)},
       {std::string("a_t"), new LiteralBool(true)}};

  // perform evaluation-based testing on original and rewritten circuit
  Ast* originalAst = generateAst(DEMO_CIRCUIT_NO);
  circuitOutputComparer(*originalAst, *astToRewrite, 934471, 5000, inputValues);
}
