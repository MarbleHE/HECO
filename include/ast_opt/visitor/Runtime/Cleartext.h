#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_

#include <vector>
#include <cstdint>

#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/Runtime/AbstractValue.h"
#include "ast_opt/ast/ExpressionList.h"

template<typename T>
class Cleartext : public AbstractValue {
 private:
  std::vector<T> data;

 public:
  explicit Cleartext(Literal<T> &node);

  explicit Cleartext(ExpressionList &node);

  const std::vector<T> &getData() const;
};

template<typename>
struct isCleartext : public std::false_type {};

template<typename T>
struct isCleartext<Cleartext<T>> : public std::true_type {};

template<typename T>
Cleartext<T>::Cleartext(Literal<T> &node) {
  data.push_back(node.getValue());
}

template<typename T>
Cleartext<T>::Cleartext(ExpressionList &node) {
  auto expressions = node.getExpressions();
  for (auto &expression : expressions) {
    if (auto exprCasted = dynamic_cast<Literal<T> *>(&expression.get())) {
      data.push_back(exprCasted->getValue());
    } else {
      throw std::runtime_error("");
    }
  }
}

template<typename T>
const std::vector<T> &Cleartext<T>::getData() const {
  return data;
}

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
