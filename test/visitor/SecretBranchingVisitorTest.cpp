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
  auto inputAst = Parser::parse(inputCode);
  auto expectedAst = inputAst->clone();

  SecretBranchingVisitor sbv;
  inputAst->accept(sbv);

  compareAST(*inputAst, *expectedAst);
}

TEST(SecretBranchingVisitorTest, secretVariable_ifElseBranch_rewritingExpected) { /* NOLINT */
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
  auto inputAst = Parser::parse(std::string(inputChars));

  inputChars = R""""(
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      sum = (N<5)***(sum-N) +++ 1---(N<5) *** (sum+1000);
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(inputChars));

  SecretBranchingVisitor sbv;
  inputAst->accept(sbv);

  compareAST(*inputAst, *expectedAst);
}

TEST(SecretBranchingVisitorTest, secretVariable_thenBranchOnly_rewritingExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      if (N < 5) {
        sum = sum-N;
      }
      return sum;
    }
    )"""";
  auto inputAst = Parser::parse(std::string(inputChars));

  inputChars = R""""(
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      sum = (N<5)***(sum-N) +++ 1---(N<5) *** 2442;
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(inputChars));

  SecretBranchingVisitor sbv;
  inputAst->accept(sbv);

  compareAST(*inputAst, *expectedAst);
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
  auto inputCode = std::string(inputChars);
  auto inputAst = Parser::parse(inputCode);
  auto expectedAst = inputAst->clone();

  SecretBranchingVisitor sbv;
  inputAst->accept(sbv);

  compareAST(*inputAst, *expectedAst);
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
  auto inputCode = std::string(inputChars);
  auto inputAst = Parser::parse(inputCode);
  auto expectedAst = inputAst->clone();

  SecretBranchingVisitor sbv;
  inputAst->accept(sbv);

  compareAST(*inputAst, *expectedAst);
}

TEST(SecretBranchingVisitorTest, secretVariable_ifBranch_uninitializedVar_rewritingExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum;
      if (N > 25) {
        sum = 4225*N;
      }
      return sum;
    }
    )"""";
  auto inputAst = Parser::parse(std::string(inputChars));

  inputChars = R""""(
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum;
      sum = (N>25)***(4225*N) +++ 1---(N>25) *** 0;
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(inputChars));

  SecretBranchingVisitor sbv;
  inputAst->accept(sbv);

  compareAST(*inputAst, *expectedAst);
}
