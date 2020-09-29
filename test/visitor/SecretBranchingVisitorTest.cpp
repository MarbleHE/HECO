#include <test/ASTComparison.h>
#include <include/ast_opt/visitor/SecretBranchingVisitor.h>
#include "include/ast_opt/visitor/ControlFlowGraph/ControlFlowGraphVisitor.h"
#include "include/ast_opt/parser/Parser.h"
#include "gtest/gtest.h"

TEST(SecretBranchingVisitorTest, nonSecretVariable_ifStmt_noRemovalExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(int N) {
      int sum = 2442;
      if (N < 5) {
        sum = sum-N;
      }
      return sum;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(inputCode, createdNodes);
  auto expectedAst = inputAst->clone();

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(3).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

TEST(SecretBranchingVisitorTest, DISABLED_secretVariable_ifElseBranch_rewritingExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      if (N < 5) {
        sum = sum-N;
      } else {
        sum = sum + 1000;
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  const char *expectedChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      sum = (N<5)***(sum-N) +++ 1---(N<5) *** (sum+1000);
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(expectedChars));

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(3).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

TEST(SecretBranchingVisitorTest, DISABLED_secretVariable_thenBranchOnly_rewritingExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      if (N < 5) {
        sum = sum-N;
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  const char *expectedChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      sum = (N<5)***(sum-N) +++ 1---(N<5) *** 2442;
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(expectedChars));

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(3).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

TEST(SecretBranchingVisitorTest, secretVariable_ifStmt_unsupportedBodyIf_noRemovalExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      if (N < 5) {
        for (int i = 0; i < 100; i = i + 1) {
          sum = sum + i;
        }
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputCode = std::string(inputChars);
  auto inputAst = Parser::parse(inputCode, createdNodes);
  auto expectedAst = inputAst->clone();

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(3).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

TEST(SecretBranchingVisitorTest, secretVariable_ifStmt_unsupportedBodyReturn_noRemovalExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      if (N < 5) {
        return 0;
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputCode = std::string(inputChars);
  auto inputAst = Parser::parse(inputCode, createdNodes);
  auto expectedAst = inputAst->clone();

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(3).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

TEST(SecretBranchingVisitorTest, DISABLED_secretVariable_ifBranch_uninitializedVar_rewritingExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum;
      if (N > 25) {
        sum = 4225*N;
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  const char *expectedChars = R""""(
    public int main(secret int N) {
      int sum;
      sum = (N>25)***(4225*N) +++ 1---(N>25) *** 0;
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(expectedChars));

  // get binary expression node (N > 25)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(2).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}
