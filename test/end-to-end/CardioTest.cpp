#include <random>
#include <include/heco/ast_parser/Parser.h>
#include "heco/ast/AbstractNode.h"
#include "gtest/gtest.h"

class CardioTest : public ::testing::Test
{ /* NOLINT (predictable sequence expected) */
protected:
  void SetUp() override
  {
  }

  std::unique_ptr<AbstractNode> getInputs(size_t /* size */)
  {
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

  static std::unique_ptr<AbstractNode> getOutputs()
  {
    const char *outputs = R""""(
      riskScore = r;
    )"""";
    return Parser::parse(std::string(outputs));
  }

  std::unique_ptr<AbstractNode> getEvaluationProgram(std::vector<std::reference_wrapper<AbstractNode>> &createdNodes)
  {
    // TODO: Implement cardio program
    // program's input
    const char *inputs = R""""(
    )"""";
    return Parser::parse(std::string(inputs), createdNodes);
  }
};

int computeCardioRiskScore()
{
  return 0;
  // TODO: Implement me
}

TEST_F(CardioTest, DISABLED_originalProgramTest)
{ /* NOLINT */
  // TODO: Implement me
}