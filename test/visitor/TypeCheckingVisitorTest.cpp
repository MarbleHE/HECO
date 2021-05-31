#include "ast_opt/visitor/controlFlowGraph/ControlFlowGraphVisitor.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/parser/Parser.h"
#include "gtest/gtest.h"

TEST(TypeCheckingVisitorTest, simpleSecretTypeRecognition) { /* NOLINT */
  const char *inputChars = R""""(
    public secret int main(int N) {
      secret int sum = 2442;
      double k = 1.23332;
      return sum;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  TypeCheckingVisitor tcv;
  inputAST->begin()->accept(tcv);

  auto &rootScope = tcv.getRootScope();

  auto itFunctionChildren = inputAST->begin()->begin();
  std::advance(itFunctionChildren, 1);
  auto &functionBlockScope = rootScope.getNestedScopeByCreator(*itFunctionChildren);

  ScopedIdentifier &variableN = functionBlockScope.resolveIdentifier("N");
  auto variableN_datatype = tcv.getVariableDatatype(const_cast<ScopedIdentifier &>(variableN));
  EXPECT_EQ(variableN_datatype.getType(), Type::INT);
  EXPECT_FALSE(variableN_datatype.getSecretFlag());

  ScopedIdentifier variableSum = functionBlockScope.resolveIdentifier("sum");
  auto variableSum_datatype = tcv.getVariableDatatype(variableSum);
  EXPECT_EQ(variableSum_datatype.getType(), Type::INT);
  EXPECT_TRUE(variableSum_datatype.getSecretFlag());

  ScopedIdentifier variableK = functionBlockScope.resolveIdentifier("k");
  auto variableK_datatype = tcv.getVariableDatatype(variableK);
  EXPECT_EQ(variableK_datatype.getType(), Type::DOUBLE);
  EXPECT_FALSE(variableK_datatype.getSecretFlag());
}

TEST(TypeCheckingVisitorTest, incompatibleTypes) { /* NOLINT */
  const char *inputChars = R""""(
    public secret int main(int N) {
      secret int sum = 2442;
      double k = 1.23332;
      return sum*k;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  TypeCheckingVisitor tcv;
  EXPECT_THROW(inputAST->begin()->accept(tcv), std::runtime_error);
}

TEST(TypeCheckingVisitorTest, invalidIndexAccessType) { /* NOLINT */
  const char *inputChars = R""""(
    public secret int main(int N) {
      int options = {2442, 2, 4, 14, 2};
      return sum[2.0];
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  EXPECT_THROW(inputAST->begin()->accept(tcv), std::runtime_error);
}

TEST(TypeCheckingVisitorTest, binaryExpressionDatatype) { /* NOLINT */
  const char *inputChars = R""""(
    public secret int main(int N) {
      secret int sum = 2442;
      return 4*sum;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  inputAST->begin()->accept(tcv);

  auto binaryExprDatatype = tcv.getExpressionDatatype(
      dynamic_cast<AbstractExpression &>(createdNodes.at(6).get()));
  EXPECT_EQ(binaryExprDatatype.getType(), Type::INT);
  EXPECT_FALSE(binaryExprDatatype.getSecretFlag());
}

TEST(TypeCheckingVisitorTest, deepNestedBinaryExpressionDatatype) { /* NOLINT */
  const char *inputChars = R""""(
    public secret int main(int N, int M) {
      secret int sum = (4096 - (2442 * N)) + (M * 4);
      return sum;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  inputAST->begin()->accept(tcv);

  auto binExpr1 = tcv.getExpressionDatatype(dynamic_cast<AbstractExpression &>(createdNodes.at(6).get()));
  EXPECT_EQ(binExpr1.getType(), Type::INT);
  EXPECT_FALSE(binExpr1.getSecretFlag());
  auto binExpr2 = tcv.getExpressionDatatype(dynamic_cast<AbstractExpression &>(createdNodes.at(7).get()));
  EXPECT_EQ(binExpr2.getType(), Type::INT);
  EXPECT_FALSE(binExpr2.getSecretFlag());
  auto binExpr3 = tcv.getExpressionDatatype(dynamic_cast<AbstractExpression &>(createdNodes.at(10).get()));
  EXPECT_EQ(binExpr3.getType(), Type::INT);
  EXPECT_FALSE(binExpr3.getSecretFlag());
  auto binExpr4 = tcv.getExpressionDatatype(dynamic_cast<AbstractExpression &>(createdNodes.at(11).get()));
  EXPECT_EQ(binExpr4.getType(), Type::INT);
  EXPECT_FALSE(binExpr4.getSecretFlag());
}

TEST(TypeCheckingVisitorTest, unaryExpressionDatatype) { /* NOLINT */
  const char *inputChars = R""""(
    public secret bool main(bool isRecommended) {
      secret bool b = !isRecommended;
      return b;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  inputAST->begin()->accept(tcv);

  auto binaryExprDatatype = tcv.getExpressionDatatype(
      dynamic_cast<AbstractExpression &>(createdNodes.at(3).get()));
  EXPECT_EQ(binaryExprDatatype.getType(), Type::BOOL);
  EXPECT_FALSE(binaryExprDatatype.getSecretFlag());
}

TEST(TypeCheckingVisitorTest, returnTypeNotMatchingSpecifiedType) { /* NOLINT */
  const char *inputChars = R""""(
    public secret int main(bool isRecommended) {
      secret bool b = !isRecommended;
      return b;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  EXPECT_THROW(inputAST->begin()->accept(tcv), std::runtime_error);
}

TEST(TypeCheckingVisitorTest, returnTypeNonVoidButNoReturnStatementGiven) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(bool isRecommended) {
      secret bool b = !isRecommended;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  EXPECT_THROW(inputAST->begin()->accept(tcv), std::runtime_error);
}

TEST(TypeCheckingVisitorTest, returnTypeNotMatchingSpecifiedSecretness) { /* NOLINT */
  const char *inputChars = R""""(
    public bool main(bool isRecommended) {
      secret bool b = !isRecommended;
      return b;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  EXPECT_THROW(inputAST->begin()->accept(tcv), std::runtime_error);
}

TEST(TypeCheckingVisitorTest, returnTypeVoidButReturningValue) { /* NOLINT */
  const char *inputChars = R""""(
    public void main(bool isRecommended) {
      return b;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  EXPECT_THROW(inputAST->begin()->accept(tcv), std::runtime_error);
}

TEST(TypeCheckingVisitorTest, secretTainting_ifCondition) { /* NOLINT */
  const char *inputChars = R""""(
    public secret int main(int threshold) {
      secret int val = 2411;
      if (val < threshold) {
        return 1;
      }
      return 0;
    }
    )"""";
  auto inputCode = std::string(inputChars);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAST = Parser::parse(inputCode, createdNodes);

  TypeCheckingVisitor tcv;
  inputAST->begin()->accept(tcv);

  auto binaryExpressionNodeId = createdNodes.at(6).get().getUniqueNodeId();
  EXPECT_TRUE(tcv.isSecretTaintedNode(binaryExpressionNodeId));
}

TEST(TypeCheckingVisitorTest, overwriteSecret) { /* NOLINT */
    const char *inputChars = R""""(
    public secret int main(int N) {
      secret int sum = 2442;
      sum = 5555;
      return sum + 1;
    }
    )"""";
    auto inputCode = std::string(inputChars);
    std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
    auto inputAST = Parser::parse(inputCode, createdNodes);

    TypeCheckingVisitor tcv;
    inputAST->begin()->accept(tcv);

    auto binaryExpressionNodeId = createdNodes.at(9).get().getUniqueNodeId();

    //TODO: Currently, the implementation says that (sum + 1) is tainted
    // However, it might be interesting to allow "overwriting" secrets with public values
    // and considering this in the secret-tainting analysis, as described in https://github.com/MarbleHE/ABC/issues/5
    EXPECT_TRUE(tcv.isSecretTaintedNode(binaryExpressionNodeId));
}
