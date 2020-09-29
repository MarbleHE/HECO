#include "include/ast_opt/visitor/ControlFlowGraph/ControlFlowGraphVisitor.h"
#include "include/ast_opt/visitor/TypeCheckingVisitor.h"
#include "include/ast_opt/ast/Function.h"
#include "include/ast_opt/parser/Parser.h"
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
  auto &variableN_datatype = tcv.getVariableDatatype(const_cast<ScopedIdentifier &>(variableN));
  ASSERT_EQ(variableN_datatype.getType(), Type::INT);
  ASSERT_FALSE(variableN_datatype.getSecretFlag());

  ScopedIdentifier variableSum = functionBlockScope.resolveIdentifier("sum");
  auto &variableSum_datatype = tcv.getVariableDatatype(variableSum);
  ASSERT_EQ(variableSum_datatype.getType(), Type::INT);
  ASSERT_TRUE(variableSum_datatype.getSecretFlag());

  ScopedIdentifier variableK = functionBlockScope.resolveIdentifier("k");
  auto &variableK_datatype = tcv.getVariableDatatype(variableK);
  ASSERT_EQ(variableK_datatype.getType(), Type::DOUBLE);
  ASSERT_FALSE(variableK_datatype.getSecretFlag());
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
