#ifndef AST_OPTIMIZER_INCLUDE_AST_OPERATOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPERATOR_H_

#include <utility>
#include <variant>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include "ast_opt/visitor/Visitor.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/LiteralString.h"
#include "ast_opt/ast/LiteralBool.h"
#include "ast_opt/ast/LiteralInt.h"
#include "ast_opt/ast/OpSymbEnum.h"
#include "ast_opt/ast/LiteralFloat.h"

class Operator : public AbstractNode {
 private:
  std::string operatorString;
  OpSymbolVariant operatorSymbol;

 public:
  explicit Operator(LogCompOp op);

  explicit Operator(ArithmeticOp op);

  explicit Operator(UnaryOp op);

  explicit Operator(OpSymbolVariant op);

  [[nodiscard]] const std::string &getOperatorString() const;

  [[nodiscard]] const OpSymbolVariant &getOperatorSymbol() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  bool isUndefined();

  bool operator==(const Operator &rhs) const;

  bool operator!=(const Operator &rhs) const;

  [[nodiscard]] bool equals(OpSymbolVariant op) const;

  AbstractLiteral *applyOperator(AbstractLiteral *lhs, AbstractLiteral *rhs);

  AbstractLiteral *applyOperator(AbstractLiteral *rhs);

  template<typename A>
  AbstractLiteral *applyOperator(A *lhs, AbstractLiteral *rhs);

  AbstractLiteral *applyOperator(LiteralInt *lhs, LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralBool *lhs, LiteralBool *rhs);

  AbstractLiteral *applyOperator(LiteralInt *lhs, LiteralBool *rhs);

  AbstractLiteral *applyOperator(LiteralBool *lhs, LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralString *lhs, LiteralString *rhs);

  static AbstractLiteral *applyOperator(LiteralBool *lhs, LiteralString *rhs);

  static AbstractLiteral *applyOperator(LiteralString *lhs, LiteralBool *rhs);

  static AbstractLiteral *applyOperator(LiteralInt *lhs, LiteralString *rhs);

  static AbstractLiteral *applyOperator(LiteralString *lhs, LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralBool *rhs);

  AbstractLiteral *applyOperator(LiteralString *rhs);

  AbstractLiteral *applyOperator(LiteralFloat *lhs, LiteralFloat *rhs);

  AbstractLiteral *applyOperator(LiteralFloat *lhs, LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralInt *lhs, LiteralFloat *rhs);

  AbstractLiteral *applyOperator(LiteralFloat *lhs, LiteralBool *rhs);

  AbstractLiteral *applyOperator(LiteralBool *lhs, LiteralFloat *rhs);

  static AbstractLiteral *applyOperator(LiteralFloat *lhs, LiteralString *rhs);

  static AbstractLiteral *applyOperator(LiteralString *lhs, LiteralFloat *rhs);

  AbstractLiteral *applyOperator(LiteralFloat *rhs);

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool supportsCircuitMode() override;

  ~Operator() override;

  Operator *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] bool equals(ArithmeticOp op) const;

  [[nodiscard]] bool equals(LogCompOp op) const;

  [[nodiscard]] bool equals(UnaryOp op) const;

  [[nodiscard]] bool isArithmeticOp() const;

  [[nodiscard]] bool isLogCompOp() const;

  [[nodiscard]] bool isUnaryOp() const;

  template<typename R, typename T>
  R acc(std::function<R(T, T)> func, std::vector<T> operands) {
    auto it = operands.begin();
    // result = operands[0] ⊕ operands[1]
    R result = func(*it, *(++it));
    if (operands.size() > 2) {
      // result = ((((result ⊕ operands[2]) ⊕ operands[3]) ⊕ ...) ⊕ operands[N])
      for (++it; it!=operands.end(); ++it) result = func(result, *it);
    }
    return result;
  }

  template<typename T>
  bool appPairwise(std::function<bool(T, T)> func, std::vector<T> operands) {
    auto it = operands.begin();
    // result = operands[0] ⊕ operands[1]
    bool result = func(*it, *(++it));
    if (operands.size() > 2) {
      // result = ((result && (operands[1] ⊕ operands[2])) && (operands[2] ⊕ operands[3])) ...
      for (; it!=operands.end() - 1; ++it) result = result && func(*it, *(it + 1));
    }
    return result;
  }

  template<typename abstractType, typename primitiveType>
  std::vector<primitiveType> convert(std::vector<AbstractLiteral *> operands) {
    std::vector<primitiveType> vec;
    std::transform(operands.begin(), operands.end(), std::back_inserter(vec),
                   [](AbstractLiteral *lit) { return lit->castTo<abstractType>()->getValue(); });
    return vec;
  }

  AbstractLiteral *applyOperator(std::vector<AbstractLiteral *> operands);

  AbstractLiteral *applyOperator(std::vector<int> operands);

  AbstractLiteral *applyOperator(std::vector<float> operands);

  AbstractLiteral *applyOperator(std::vector<bool> operands);

  AbstractLiteral *applyOperator(std::vector<std::string> operands);

  [[nodiscard]] bool isCommutative();

  [[nodiscard]] bool isLeftAssociative();

  /// An operator is considered as non-partial evaluable if its not possible to apply the operator only on a few
  /// operands and then store the intermediate result to continue evaluation after more operands are known.
  /// For example, 7 + a + 21 + 9 is partially evaluable to a + 37.
  /// But the expression 32 < 193 < a < 12 would partially evaluate 32 < 193 = true and then lead to true < a < 12 which
  /// is obviously not correct. Because of that we require that all operands are known for these kind of operators.
  [[nodiscard]] bool supportsPartialEvaluation();
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPERATOR_H_
