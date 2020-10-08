#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_

#include <vector>
#include <cstdint>

#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/runtime/AbstractValue.h"
#include "ast_opt/ast/ExpressionList.h"

template<typename Derived, typename Base, typename Del>
std::unique_ptr<Derived, Del> static_unique_ptr_cast(std::unique_ptr<Base, Del> &&p) {
  auto d = static_cast<Derived *>(p.release());
  return std::unique_ptr<Derived, Del>(d, std::move(p.get_deleter()));
}

/// This class serves as a common base class for all Cleartext<T> classes
class ICleartext : public AbstractValue {
 protected:
  /// Protected constructor: makes sure that class is abstract, i.e., cannot be instantiated.
  ICleartext() = default;

 public:
  /// Default destructor.
  ~ICleartext() override = default;

  virtual std::unique_ptr<ICleartext> clone() = 0;
};

template<typename T>
class Cleartext : public ICleartext {
 private:
  std::vector<T> data;

 public:
  explicit Cleartext(Literal<T> &node);

  explicit Cleartext(ExpressionList &node);

  explicit Cleartext(std::vector<std::unique_ptr<ICleartext>> &cleartexts);

  // copy constructor
  Cleartext(const Cleartext<T> &other) {
    data = other.getData();
  }

  const std::vector<T> &getData() const;

 private:
  std::unique_ptr<ICleartext> clone() override;
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
Cleartext<T>::Cleartext(std::vector<std::unique_ptr<ICleartext>> &cleartexts) {
  // go through each ICleartext in this vector
  for (std::unique_ptr<ICleartext> &cleartextUniquePtrRef : cleartexts) {
    // if this is a Cleartext<T> of the same type
    if (auto otherCleartext = dynamic_cast<Cleartext<T> *>(cleartextUniquePtrRef.get())) {
      // merge data from the other cleartext into this cleartext
      data.insert(data.end(), otherCleartext->data.begin(), otherCleartext->data.end());
    } else {
      throw std::runtime_error("Cannot create Cleartext<T> of multiple other Cleartext<T> with different types!");
    }
  }
}

template<typename T>
const std::vector<T> &Cleartext<T>::getData() const {
  return data;
}

template<typename T>
std::unique_ptr<ICleartext> Cleartext<T>::clone() {
  auto cloned = std::make_unique<Cleartext<T>>(*this);
  return cloned;
}

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
