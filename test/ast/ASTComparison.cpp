#include "abc/ast/Assignment.h"
#include "abc/ast/BinaryExpression.h"
#include "abc/ast/Block.h"
#include "abc/ast/Call.h"
#include "abc/ast/ExpressionList.h"
#include "abc/ast/For.h"
#include "abc/ast/Function.h"
#include "abc/ast/FunctionParameter.h"
#include "abc/ast/If.h"
#include "abc/ast/IndexAccess.h"
#include "abc/ast/Literal.h"
#include "abc/ast/OperatorExpression.h"
#include "abc/ast/Return.h"
#include "abc/ast/UnaryExpression.h"
#include "abc/ast/Variable.h"
#include "abc/ast/VariableDeclaration.h"
#include "ASTComparison.h"

::testing::AssertionResult compareAST(const AbstractNode &ast1, const AbstractNode &ast2) {
  {
    if (typeid(ast1)!=typeid(ast2)) {
      return ::testing::AssertionFailure() << "AST nodes have different types: " << ast1.toString(false) << " vs "
                                           << ast2.toString(false);
    } else if (typeid(ast1)==typeid(const Assignment &)) {
      // No non-AST attributes
    } else if (typeid(ast1)==typeid(const BinaryExpression &)) {
      auto b1 = dynamic_cast<const BinaryExpression &>(ast1);
      auto b2 = dynamic_cast<const BinaryExpression &>(ast2);
      if (b1.getOperator().toString()!=b2.getOperator().toString()) {
        return ::testing::AssertionFailure() << "BinaryExpressions nodes have different operators: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(Block)) {
      // No non-AST attributes
    } else if (typeid(ast1)==typeid(ExpressionList)) {
      // No non-AST attributes
    } else if (typeid(ast1)==typeid(For)) {
      // No non-AST attributes
    } else if (typeid(ast1)==typeid(const Function &)) {
      auto f1 = dynamic_cast<const Function &>(ast1);
      auto f2 = dynamic_cast<const Function &>(ast2);
      if (f1.getIdentifier()!=f2.getIdentifier()) {
        return ::testing::AssertionFailure() << "Function nodes have different identifiers: " << ast1.toString(false)
                                             << " vs " << ast2.toString(false);
      }
      if (f1.getReturnType()!=f2.getReturnType()) {
        return ::testing::AssertionFailure() << "Function nodes have different return type: " << ast1.toString(false)
                                             << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const FunctionParameter &)) {
      auto f1 = dynamic_cast<const FunctionParameter &>(ast1);
      auto f2 = dynamic_cast<const FunctionParameter &>(ast2);
      if (f1.getIdentifier()!=f2.getIdentifier()) {
        return ::testing::AssertionFailure() << "FunctionParameter nodes have different identifiers: "
                                             << ast1.toString(false)
                                             << " vs " << ast2.toString(false);
      }
      if (f1.getParameterType()!=f2.getParameterType()) {
        return ::testing::AssertionFailure() << "FunctionParameter nodes have different parameter type: "
                                             << ast1.toString(false)
                                             << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const If &)) {
      // No non-AST attributes
    } else if (typeid(ast1)==typeid(IndexAccess)) {
      // No non-AST attributes
    } else if (typeid(ast1)==typeid(const OperatorExpression &)) {
      auto o1 = dynamic_cast<const OperatorExpression &>(ast1);
      auto o2 = dynamic_cast<const OperatorExpression &>(ast2);
      if (o1.getOperator().toString()!=o2.getOperator().toString()) {
        return ::testing::AssertionFailure() << "OperatorExpression nodes have different operators: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(Return)) {
      // No non-AST attributes
    } else if (typeid(ast1)==typeid(const UnaryExpression &)) {
      auto u1 = dynamic_cast<const UnaryExpression &>(ast1);
      auto u2 = dynamic_cast<const UnaryExpression &>(ast2);
      if (u1.getOperator().toString()!=u2.getOperator().toString()) {
        return ::testing::AssertionFailure() << "UnaryExpression nodes have different operators: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const Variable &)) {
      auto v1 = dynamic_cast<const Variable &>(ast1);
      auto v2 = dynamic_cast<const Variable &>(ast2);
      if (v1.getIdentifier()!=v2.getIdentifier()) {
        return ::testing::AssertionFailure() << "Variable nodes have different identifiers: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const VariableDeclaration &)) {
      auto v1 = dynamic_cast<const VariableDeclaration &>(ast1);
      auto v2 = dynamic_cast<const VariableDeclaration &>(ast2);
      if (v1.getDatatype()!=v2.getDatatype()) {
        return ::testing::AssertionFailure() << "VariableDeclaration nodes have different datatypes: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const LiteralBool &)) {
      auto l1 = dynamic_cast<const LiteralBool &>(ast1);
      auto l2 = dynamic_cast<const LiteralBool &>(ast2);
      if (l1.getValue()!=l2.getValue()) {
        return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const LiteralChar &)) {
      auto l1 = dynamic_cast<const LiteralChar &>(ast1);
      auto l2 = dynamic_cast<const LiteralChar &>(ast2);
      if (l1.getValue()!=l2.getValue()) {
        return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const LiteralInt &)) {
      auto l1 = dynamic_cast<const LiteralInt &>(ast1);
      auto l2 = dynamic_cast<const LiteralInt &>(ast2);
      if (l1.getValue()!=l2.getValue()) {
        return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const LiteralFloat &)) {
      auto l1 = dynamic_cast<const LiteralFloat &>(ast1);
      auto l2 = dynamic_cast<const LiteralFloat &>(ast2);
      if (l1.getValue()!=l2.getValue()) {
        return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const LiteralDouble &)) {
      auto l1 = dynamic_cast<const LiteralDouble &>(ast1);
      auto l2 = dynamic_cast<const LiteralDouble &>(ast2);
      if (l1.getValue()!=l2.getValue()) {
        return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const LiteralString &)) {
      auto l1 = dynamic_cast<const LiteralString &>(ast1);
      auto l2 = dynamic_cast<const LiteralString &>(ast2);
      if (l1.getValue()!=l2.getValue()) {
        return ::testing::AssertionFailure() << "Literal nodes have different values: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else if (typeid(ast1)==typeid(const Call &)) {
      auto l1 = dynamic_cast<const Call &>(ast1);
      auto l2 = dynamic_cast<const Call &>(ast2);

      auto l1Args = l1.getArguments();
      bool equalArgs = std::equal(l1Args.begin(), l1Args.end(), l2.getArguments().begin(),
                                  [&](const std::reference_wrapper<AbstractExpression> &ae1,
                                      const std::reference_wrapper<AbstractExpression> &ae2) {
                                    auto comp = compareAST(ae1.get(), ae2.get());
                                    return comp==::testing::AssertionSuccess();
                                  });

      if (l1.getIdentifier()!=l2.getIdentifier() || !equalArgs) {
        return ::testing::AssertionFailure() << "Call nodes have different values: "
                                             << ast1.toString(false) << " vs " << ast2.toString(false);
      }
    } else {
      throw std::runtime_error("Something bad happened while comparing ASTs.");
    }

    // Compare Children
    if (ast1.countChildren()!=ast2.countChildren())
      return ::testing::AssertionFailure() << "Nodes do not have equal number of children!" << ast1.toString(false) << " has " << ast1.countChildren() << " while " << ast2.toString(false) << " has " << ast2.countChildren();
    auto it1 = ast1.begin();
    auto it2 = ast2.begin();
    for (; it1!=ast1.end() && it2!=ast2.end(); ++it1, ++it2) {
      auto r = compareAST(*it1, *it2);
      if (!r) {
        return ::testing::AssertionFailure() << ast1.toString(true) << " and " << ast2.toString(true)
                                             << " differ in children: " << it1->toString(false) << " vs "
                                             << it2->toString(false)
                                             << "Original issue:" << r.message();
      }
    }
    return ::testing::AssertionSuccess();
  }
}
