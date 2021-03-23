#include "test/ASTComparison.h"
#include "include/ast_opt/visitor/SecretBranchingVisitor.h"
#include "include/ast_opt/visitor/controlFlowGraph/ControlFlowGraphVisitor.h"
#include "include/ast_opt/parser/Parser.h"
#include "gtest/gtest.h"

// ================================================
// ===== Tests with expected rewriting ============
// ================================================


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
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  const char *expectedChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      sum = ((N<5)***(sum-N)) +++ ((1---(N<5)) *** (sum+1000));
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(expectedChars));

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(6).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
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
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  const char *expectedChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      sum = ((N<5)***(sum-N)) +++ ((1---(N<5)) *** 2442);
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(expectedChars));

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(6).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

TEST(SecretBranchingVisitorTest, secretVariable_NestedThenBranch_rewritingExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N, secret int M) {
      int sum = 2442;
      if (N < 5) {
        if (M > 0) {
          sum = sum-N;
        }
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  const char *expectedChars = R""""(
    public int main(secret int N, secret int M) {
      int sum = 2442;
      sum = ((N<5)***(((M>0)***(sum-N)) +++ ((1---(M>0))***2442))) +++ ((1---(N<5))***2442);
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(expectedChars));

  // get binary expression nodes
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(7).get().getUniqueNodeId(), true);
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(10).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
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
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  const char *expectedChars = R""""(
    public int main(secret int N) {
      int sum;
      sum = ((N>25)***(4225*N)) +++ ((1---(N>25)) *** sum);
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(expectedChars));

  // get binary expression node (N > 25)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(5).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

// ================================================
// ===== Tests with no expected rewriting =========
// ================================================

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
      createdNodes.at(6).get().getUniqueNodeId(), false);

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
      createdNodes.at(6).get().getUniqueNodeId(), true);

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
      createdNodes.at(6).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}


TEST(SecretBranchingVisitorTest, secretVariable_ifBranch_unsupportedBodyFor_noRewritingExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum;
      if (N > 25) {
        for (int i = 0; i < 225; i = i + 1) {
          sum = sum + N;
        }
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  auto expectedAst = inputAst->clone();

  // get binary expression node (N > 25)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(5).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

TEST(SecretBranchingVisitorTest, secretVariable_ifBranch_unsupportedBodyReturn_noRewritingExpected) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum;
      if (N > 25) {
        return sum+34;
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

  auto expectedAst = inputAst->clone();

  // get binary expression node (N > 25)
  SecretTaintedNodesMap secretTaintedNodesMap;
  secretTaintedNodesMap.insert_or_assign(
      createdNodes.at(5).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap);
  inputAst->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst, *expectedAst));
}

TEST(SecretBranchingVisitorTest, noInteractionBetweenDifferentParsings) { /* NOLINT */

  // FIRST PARSE:
  const char *inputChars1 = R""""(
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
  auto inputAst1 = Parser::parse(std::string(inputChars1), createdNodes);

  const char *expectedChars1 = R""""(
    public int main(secret int N) {
      int sum = 2442;
      sum = ((N<5)***(sum-N)) +++ ((1---(N<5)) *** (sum+1000));
      return sum;
    }
    )"""";
  auto expectedAst = Parser::parse(std::string(expectedChars1));

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap1;
  secretTaintedNodesMap1.insert_or_assign(
      createdNodes.at(6).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv(secretTaintedNodesMap1);
  inputAst1->accept(sbv);

  EXPECT_TRUE(compareAST(*inputAst1, *expectedAst));


  // SECOND PARSE:
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      if (N < 5) {
        sum = sum-N;
      }
      return sum;
    }
    )"""";
  //std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst2 = Parser::parse(std::string(inputChars), createdNodes);

  const char *expectedChars2 = R""""(
    public int main(secret int N) {
      int sum = 2442;
      sum = ((N<5)***(sum-N)) +++ ((1---(N<5)) *** 2442);
      return sum;
    }
    )"""";
  auto expectedAst2 = Parser::parse(std::string(expectedChars2));

  // get binary expression node (N < 5)
  SecretTaintedNodesMap secretTaintedNodesMap2;
  secretTaintedNodesMap2.insert_or_assign(
      createdNodes.at(6).get().getUniqueNodeId(), true);

  SecretBranchingVisitor sbv2(secretTaintedNodesMap2);
  inputAst2->accept(sbv2);

  EXPECT_TRUE(compareAST(*inputAst2, *expectedAst2));
}

TEST(SecretBranchingVisitorTest, ternaryOperator) {
  // Skipping until ternaryOperator implementation is finished
  GTEST_SKIP();

//  const char *inputChars = R""""(
//    public int main(secret int N) {
//      int sum;
//      sum =  N > 25 ? 5 : 6;
//      return sum;
//    }
//    )"""";
//  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
//  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);
//
//  // get binary expression node (N > 25)
//  SecretTaintedNodesMap secretTaintedNodesMap;
//  secretTaintedNodesMap.insert_or_assign(
//      createdNodes.at(5).get().getUniqueNodeId(), true);
//
//
//  // Instead of manually specifying target AST,
//  // we write the equivalent code and let the parser translate that into an AST
//  const char *equivalentChars = R""""(
//    public int main(secret int N) {
//      int sum;
//      if(N > 25) {
//        sum = 5;
//      } else {
//        sum = 6;
//      }
//      return sum;
//    }
//    )"""";
//  auto equivalentAst = Parser::parse(std::string(equivalentChars), createdNodes);
//
//  // get binary expression node (N > 25)
//  SecretTaintedNodesMap secretTaintedNodesMapEquiv;
//  secretTaintedNodesMapEquiv.insert_or_assign(
//      createdNodes.at(5).get().getUniqueNodeId(), true);
//  SecretBranchingVisitor equiv_sbv(secretTaintedNodesMapEquiv);
//  equivalentAst->accept(equiv_sbv);
//
//  SecretBranchingVisitor sbv(secretTaintedNodesMap);
//  inputAst->accept(sbv);
//
//  EXPECT_TRUE(compareAST(*inputAst, *equivalentAst));
}

TEST(SecretBranchingVisitorTest, no_multipleTernaryOperators) {
  // Skipping until ternaryOperator implementation is finished
  GTEST_SKIP();

//  const char *inputChars = R""""(
//    public int main(secret int N) {
//      int sum;
//      sum =  N > 25 ? ((N < 50) ? 2 : 3 ) : 6;
//      return sum;
//    }
//    )"""";
//  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
//  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);
//
//  // get binary expression node (N > 25)
//  SecretTaintedNodesMap secretTaintedNodesMap;
//  secretTaintedNodesMap.insert_or_assign(
//      createdNodes.at(5).get().getUniqueNodeId(), true);
//
//  SecretBranchingVisitor sbv(secretTaintedNodesMap);
//  EXPECT_THROW(inputAst->accept(sbv), std::runtime_error);
}


