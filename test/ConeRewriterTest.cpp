#include "AstTestingGenerator.h"
#include "ast_opt/ast/Ast.h"
#include "ast_opt/optimizer/ConeRewriter.h"
#include "ast_opt/utilities/TestUtils.h"
#include "ast_opt/utilities/DotPrinter.h"
#include "ast_opt/ast/Variable.h"
#include "gtest/gtest.h"

class ConeRewriterFixture : public ::testing::Test {
 private:
  std::unordered_map<std::string, AbstractLiteral *> *evaluationParameters;

 protected:
  ConeRewriterFixture() : evaluationParameters(nullptr) {}

  static Ast *generateAst(int demoAstId) {
    Ast *newAst = new Ast;
    AstTestingGenerator::generateAst(demoAstId, *newAst);
    return newAst;
  }

  void SetUp() override {}

  virtual void SetUp(std::unordered_map<std::string, AbstractLiteral *> &evalParams) {
    evaluationParameters = &evalParams;
  }

  bool getBoolValue(const std::string &evalParamsKey) {
    try {
      return dynamic_cast<LiteralBool *>(evaluationParameters->at(evalParamsKey))->getValue();
    } catch (std::out_of_range &exception) {
      throw std::out_of_range("");
    }
  }

  bool computeResultForCircuit19() {
    bool a11_left = getBoolValue("a_1^(1)_left");
    bool a11_right = getBoolValue("a_1^(1)_right");
    bool a21_left = getBoolValue("a_2^(1)_left");
    bool a21_right = getBoolValue("a_2^(1)_right");
    bool a12_left = getBoolValue("a_1^(2)_left");
    bool a12_right = getBoolValue("a_1^(2)_right");
    bool a22_left = getBoolValue("a_2^(2)_left");
    bool a22_right = getBoolValue("a_2^(2)_right");
    bool y1 = getBoolValue("y_1");
    bool at = getBoolValue("a_t");

    // evaluate expression that represents circuit 19 to manually verify evaluation result
    return (
        ((((a11_left & a11_right) & (a21_left ^ a21_right)) ^ ((a12_left & a12_right) & (a22_left ^ a22_right))) ^ y1)
            & at);
  }

  static void rewriteAndAssertMultiplicativeDepthDecreased(Ast *astToRewrite) {
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
    //std::cout << "Depth (before / after): (" << originalMaxMultDepth << " / " << newMaxMultDepth << ")" << std::endl;
    ASSERT_TRUE(originalMaxMultDepth > newMaxMultDepth);
  }
};

TEST_F(ConeRewriterFixture, simpleStaticConeTest) { /* NOLINT */
  const int DEMO_CIRCUIT_NO = 19;

  // -------–------------------------------------------
  // Check that the multiplicative depth decreased
  // ---------------------------------------------------

  // build demo circuit
  Ast *astToRewrite = generateAst(DEMO_CIRCUIT_NO);

  // do rewriting and check that multiplicative depth decreased
  rewriteAndAssertMultiplicativeDepthDecreased(astToRewrite);

  // -------–------------------------------------------
  // Check that the original circuit and the rewritten are logically equivalent
  // ---------------------------------------------------

  // check if rewritten circuit is logically equivalent to original one
  Ast *originalAst = generateAst(DEMO_CIRCUIT_NO);

  std::unordered_map<std::string, AbstractLiteral *> inputValues =
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
  ASSERT_EQ(*resultOriginal.front()->castTo<LiteralBool>(), LiteralBool(computeResultForCircuit19()));
  // 2. the rewriting does not modify the AST's semantics compared to the original AST (-> logical equivalence)
  ASSERT_EQ(*resultOriginal.front()->castTo<LiteralBool>(), *resultRewritten.front()->castTo<LiteralBool>());
}

TEST_F(ConeRewriterFixture, evaluationBasedTestForExtendedCircuit) { /* NOLINT */
  const int DEMO_CIRCUIT_NO = 19;

  // -------–------------------------------------------
  // Check that the multiplicative depth decreased
  // ---------------------------------------------------

  // build demo circuit
  Ast *astToRewrite = generateAst(DEMO_CIRCUIT_NO);

  // do rewriting and check that multiplicative depth decreased
  rewriteAndAssertMultiplicativeDepthDecreased(astToRewrite);

  // -------–------------------------------------------
  // Check that the original circuit and the rewritten are logically equivalent
  // ---------------------------------------------------

  // define expected input parameters and some arbitrary input values
  // (values will be overwritten by circuitOutputComparer)
  std::unordered_map<std::string, AbstractLiteral *> inputValues =
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
  Ast *originalAst = generateAst(DEMO_CIRCUIT_NO);
  circuitOutputComparer(*originalAst, *astToRewrite, 934471, 5000, inputValues);
}

TEST_F(ConeRewriterFixture, evaluationBasedTestForExtendedCircuitWithMultipleInputs) { /* NOLINT */
  const int DEMO_CIRCUIT_NO = 20;

  // -------–------------------------------------------
  // Check that the multiplicative depth decreased
  // ---------------------------------------------------

  // build demo circuit
  Ast *astToRewrite = generateAst(DEMO_CIRCUIT_NO);

  // do rewriting and check that multiplicative depth decreased
  rewriteAndAssertMultiplicativeDepthDecreased(astToRewrite);

  // -------–------------------------------------------
  // Check that the original circuit and the rewritten are logically equivalent
  // ---------------------------------------------------

  // define expected input parameters and some arbitrary input values
  // (values will be overwritten by circuitOutputComparer)
  bool anyBool = false;
  std::unordered_map<std::string, AbstractLiteral *> inputValues =
      {{std::string("a_1^(1)_left"), new LiteralBool(anyBool)},
       {std::string("a_1^(1)_right"), new LiteralBool(anyBool)},
       {std::string("a_2^(1)_left"), new LiteralBool(anyBool)},
       {std::string("a_2^(1)_right"), new LiteralBool(anyBool)},
       {std::string("a_1^(2)_left"), new LiteralBool(anyBool)},
       {std::string("a_1^(2)_right"), new LiteralBool(anyBool)},
       {std::string("a_2^(2)_left"), new LiteralBool(anyBool)},
       {std::string("a_2^(2)_right"), new LiteralBool(anyBool)},
       {std::string("y_1"), new LiteralBool(anyBool)},
       {std::string("y_2"), new LiteralBool(anyBool)},
       {std::string("y_3"), new LiteralBool(anyBool)},
       {std::string("a_t"), new LiteralBool(anyBool)}};

  // perform evaluation-based testing on original and rewritten circuit
  Ast *originalAst = generateAst(DEMO_CIRCUIT_NO);
  circuitOutputComparer(*originalAst, *astToRewrite, 886447, 4'096, inputValues);
}

TEST_F(ConeRewriterFixture, evaluationBasedTestForCircuitWithTwoCones) { /* NOLINT */
  const int DEMO_CIRCUIT_NO = 21;

  // -------–------------------------------------------
  // Check that the multiplicative depth decreased
  // ---------------------------------------------------

  // build demo circuit
  Ast *astToRewrite = generateAst(DEMO_CIRCUIT_NO);

  // do rewriting and check that multiplicative depth decreased
  rewriteAndAssertMultiplicativeDepthDecreased(astToRewrite);

  // -------–------------------------------------------
  // Check that the original circuit and the rewritten are logically equivalent
  // ---------------------------------------------------

  // define expected input parameters and some arbitrary input values
  // (values will be overwritten by circuitOutputComparer)
  bool anyBool = false;
  std::unordered_map<std::string, AbstractLiteral *> inputValues = {
      {std::string("a_1^(1)_left"), new LiteralBool(anyBool)},
      {std::string("a_1^(1)_right"), new LiteralBool(anyBool)},
      {std::string("a_2^(1)_left"), new LiteralBool(anyBool)},
      {std::string("a_2^(1)_right"), new LiteralBool(anyBool)},
      {std::string("a_1^(2)_left"), new LiteralBool(anyBool)},
      {std::string("a_1^(2)_right"), new LiteralBool(anyBool)},
      {std::string("a_2^(2)_left"), new LiteralBool(anyBool)},
      {std::string("a_2^(2)_right"), new LiteralBool(anyBool)},
      {std::string("y_1"), new LiteralBool(anyBool)},
      {std::string("y_2"), new LiteralBool(anyBool)},
      {std::string("y_3"), new LiteralBool(anyBool)},
      {std::string("y_4"), new LiteralBool(anyBool)},
      {std::string("a_t"), new LiteralBool(anyBool)},
      {std::string("b_1^(1)_left"), new LiteralBool(anyBool)},
      {std::string("b_1^(1)_right"), new LiteralBool(anyBool)},
      {std::string("b_2^(1)_left"), new LiteralBool(anyBool)},
      {std::string("b_2^(1)_right"), new LiteralBool(anyBool)},
      {std::string("z_1"), new LiteralBool(anyBool)},
      {std::string("z_2"), new LiteralBool(anyBool)},
      {std::string("z_3"), new LiteralBool(anyBool)},
      {std::string("z_4"), new LiteralBool(anyBool)},
      {std::string("b_t"), new LiteralBool(anyBool)}};

// perform evaluation-based testing on original and rewritten circuit
  Ast *originalAst = generateAst(DEMO_CIRCUIT_NO);
//  EvalPrinter evalPrinter;
//  evalPrinter.setEvaluationParameters(&inputValues)
//      .setFlagPrintEachParameterSet(true)
//      .setFlagPrintVariableHeaderOnceOnly(true)
//      .setFlagPrintEvaluationResult(true);
  circuitOutputComparer(*originalAst, *astToRewrite,
                        886447, 4'096, inputValues);
}
