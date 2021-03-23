#include <random>
#include <include/ast_opt/parser/Parser.h>
#include "ast_opt/ast/AbstractNode.h"
#include "gtest/gtest.h"

class CardioTest : public ::testing::Test {  /* NOLINT (predictable sequence expected) */
 protected:

  void SetUp() override {
  }

  std::unique_ptr<AbstractNode> getInputs(size_t /* size */) {
    const char *inputs = R""""(
      bool sex = true;
      bool antecedents = true;
      bool smoker = false;
      bool diabetes = false;
      bool high_blood_pressure = true;
      int age = 22;
      int hdl_cholesterol = 12;
      int height = 53;
      int weight = 73;
      int phy_activity = 10;
      int drinking_habits = 0;

    )"""";
    return Parser::parse(std::string(inputs));
  }

  static std::unique_ptr<AbstractNode> getOutputs() {
    const char *outputs = R""""(
      riskScore = r;
    )"""";
    return Parser::parse(std::string(outputs));
  }

  std::unique_ptr<AbstractNode> getEvaluationProgram(std::vector<std::reference_wrapper<AbstractNode>> &createdNodes) {
    // TODO: Implement cardio program
    // program's input
    const char *inputs = R""""(
      public void runKernel(int imageVec) {
        int weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
        secret int img2 = image;
        for (int x = 1; x < imgSize - 1; x = x + 1) {
          for (int y = 1; y < imgSize - 1; y = y + 1) {
            secret int value = 0;
            for (int j = -1; j < 2; j = j + 1) {
              for (int i = -1; i < 2; i = i + 1) {
                value = value + weightMatrix[i+1][j+1] * image[(x+i)*imgSize+(y+j)];
              }
            }
            img2[x*imgSize+y] = 2 * image[x*imgSize+y] - value;
          }
        }
        return;  // img2 contains result
      }
    )"""";
    return Parser::parse(std::string(inputs), createdNodes);
  }
};

int computeCardioRiskScore() {
  return 0;
  // TODO: Implement me
}

TEST_F(CardioTest, originalProgramTest) {  /* NOLINT */
  // TODO: Implement me

}

TEST_F(CardioTest, STAGE_01_typeCheckingTest) {  /* NOLINT */
  // TODO: Implement me

}

TEST_F(CardioTest, STAGE_02_ctestTest) {  /* NOLINT */
  // TODO: Implement me

}

TEST_F(CardioTest, STAGE_03_secretBranchingRemovalTest) {  /* NOLINT */
  // TODO: Implement me

}

TEST_F(CardioTest, STAGE_04_loopUnrollingTest) {  /* NOLINT */
  // TODO: Implement me

}

TEST_F(CardioTest, STAGE_05_statementVectorizationTest) {  /* NOLINT */
  // TODO: Implement me

}
