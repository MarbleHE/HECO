#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_

#include <vector>
#include <cstdint>

#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/runtime/AbstractValue.h"
#include "ast_opt/ast/ExpressionList.h"
#include "AbstractCiphertext.h"

/// This class serves as a common base class for all Cleartext<T> classes
class ICleartext : public AbstractValue {
 protected:
  /// Protected constructor: makes sure that class is abstract, i.e., cannot be instantiated.
  ICleartext() = default;

 public:
  /// Default destructor.
  ~ICleartext() override = default;

  virtual std::unique_ptr<ICleartext> clone() = 0;

  virtual std::string toString() = 0;

  virtual void setValueAtIndex(int idx, std::unique_ptr<AbstractValue> &&newValue) = 0;
};

template<typename T>
class Cleartext : public ICleartext {
 private:
  std::vector<T> data;

 public:
  explicit Cleartext(Literal<T> &node) {
    data.push_back(node.getValue());
  }

  explicit Cleartext(ExpressionList &node) {
    auto expressions = node.getExpressions();
    for (auto &expression : expressions) {
      if (auto exprCasted = dynamic_cast<Literal<T> *>(&expression.get())) {
        data.push_back(exprCasted->getValue());
      } else {
        throw std::runtime_error("Element of ExpressionList is not a " + TypeName<Literal<T>>::Get());
      }
    }
  }

  explicit Cleartext(std::unique_ptr<AbstractValue> &&abstractValue) {
    if (typeid(T)==typeid(bool)) {
      if (auto other = dynamic_cast<Cleartext<int> *>(abstractValue.get())) {
        auto dataIt = other->getData();
        data = std::vector<T>(dataIt.begin(), dataIt.end());
      } else {
        throw std::runtime_error(
            "Cannot convert given Cleartext into Cleartext<" + std::string(typeid(this).name()) + ">.");
      }
    } else {
      throw std::runtime_error("This constructor is only defined to take Cleartext<int> and generate a Cleartext<bool>.");
    }
  }

  explicit Cleartext(const std::vector<T> values) {
    data = values;
  }

  explicit Cleartext(std::vector<std::unique_ptr<ICleartext>> &cleartexts) {
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

  [[nodiscard]] bool allEqual(T value) const {
    return std::all_of(data.begin(), data.end(), [&value](const T &current) {
      return current==value;
    });
  }

  [[nodiscard]] bool allEqual() const {
    auto firstValue = data.at(0);
    return allEqual(firstValue);
  }

  // copy constructor
  Cleartext(const Cleartext<T> &other) {
    data = other.getData();
  }

  [[nodiscard]] const std::vector<T> &getData() const {
    return data;
  }

  std::unique_ptr<ICleartext> clone() override {
    auto cloned = std::make_unique<Cleartext<T>>(*this);
    return cloned;
  }

  std::string toString() override {
    std::ostringstream vts;
    std::copy(data.begin(), data.end() - 1, std::ostream_iterator<T>(vts, ", "));
    vts << data.back();
    return vts.str();
  }

  void applyBinaryOperator(std::function<T(T, T)> f, const AbstractValue &other) {
    if (auto otherCleartxt = dynamic_cast<const Cleartext<T> *>(&other)) {
      std::transform(data.begin(), data.end(), otherCleartxt->data.begin(), data.begin(), f);
    } else {
      throw std::runtime_error(
          "Given operation can only be applied on (Cleartext<T>, Cleartext<T>). "
          "This could happen, for example, if an operation is called on (Cleartext<T>, AbstractCiphertext) "
          "but the operation is unsupported in FHE.");
    }
  }

  void applyUnaryOperator(std::function<T(T)> f) {
    std::transform(data.begin(), data.end(), data.begin(), f);
  }

  void add_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::plus<T>(), other);
  }

  void subtract_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::minus<T>(), other);
  }

  void multiply_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::multiplies<T>(), other);
  }

  void divide_inplace(const AbstractValue &other) override {
    if constexpr (std::is_same<T, bool>::value) {
      throw std::invalid_argument("Cannot divide_inplace booleans. " + std::string(typeid(other).name()));
    } else {
      applyBinaryOperator(std::divides<T>(), other);
    }
  }

  void modulo_inplace(const AbstractValue &other) override {
    if constexpr (std::is_same<T, bool>::value) {
      throw std::invalid_argument("Cannot modulo_inplace booleans. " + std::string(typeid(other).name()));
    } else {
      applyBinaryOperator(std::modulus<T>(), other);
    }
  }

  void logicalAnd_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::logical_and<T>(), other);
  }

  void logicalOr_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::logical_or<T>(), other);
  }

  void logicalLess_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::less<T>(), other);
  }

  void logicalLessEqual_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::less_equal<T>(), other);
  }

  void logicalGreater_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::greater<T>(), other);
  }

  void logicalGreaterEqual_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::greater_equal<T>(), other);
  }

  void logicalEqual_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::equal_to<T>(), other);
  }

  void logicalNotEqual_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::not_equal_to<T>(), other);
  }

  void bitwiseAnd_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::bit_and<T>(), other);
  }

  void bitwiseXor_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::bit_xor<T>(), other);
  }

  void bitwiseOr_inplace(const AbstractValue &other) override {
    applyBinaryOperator(std::bit_or<T>(), other);
  }

  void logicalNot_inplace() override {
    applyUnaryOperator(std::logical_not<T>());
  }

  void bitwiseNot_inplace() override {
    if constexpr (std::is_same<T, bool>::value) {
      applyUnaryOperator([](bool b){return !b;});
    } else {
      applyUnaryOperator(std::bit_not<T>());
    }
  }

  void setValueAtIndex(int idx, std::unique_ptr<AbstractValue> &&newValue) override {
    if (auto d = dynamic_cast<Cleartext<T> *>(newValue.get())) {
      if (!d->allEqual()) {
        throw std::runtime_error("Cannot assign multiple values to a single Cleartext element.");
      }
      data[idx] = d->getData().at(0);
    } else {
      throw std::runtime_error(
          "Assigning a value to a Cleartext<T> requires the value to be a Cleartext<T> too (i.e., same type T).");
    }
  }
};

template<typename>
struct isCleartext : public std::false_type {};

template<typename T>
struct isCleartext<Cleartext<T>> : public std::true_type {};

// partial specializations of Cleartext<T> member functions

template<>
inline void Cleartext<std::string>::subtract_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply subtraction to operands of type (std::string, std:string).");
}

template<>
inline void Cleartext<std::string>::multiply_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply multiplication to operands of type (std::string, std:string).");
}

template<>
inline void Cleartext<std::string>::divide_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot division to operands of type (std::string, std:string).");
}

template<>
inline void Cleartext<float>::modulo_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply modulo_inplace to operands of type (float, float).");
}

template<>
inline void Cleartext<double>::modulo_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply modulo_inplace to operands of type (double, double).");
}

template<>
inline void Cleartext<std::string>::modulo_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply modulo_inplace to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalAnd_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply AND to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalOr_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply OR to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalLess_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply less to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalLessEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply less-equal to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalGreater_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply greater to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalGreaterEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply greater-equal to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply equal to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalNotEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply not-equal to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::bitwiseAnd_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-AND to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<float>::bitwiseAnd_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-AND to operands of type (float, float).");
}

template<>
inline void Cleartext<double>::bitwiseAnd_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-AND to operands of type (double, double).");
}

template<>
inline void Cleartext<std::string>::bitwiseOr_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-OR to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<float>::bitwiseOr_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-OR to operands of type (float, float).");
}

template<>
inline void Cleartext<double>::bitwiseOr_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-OR to operands of type (double, double).");
}

template<>
inline void Cleartext<std::string>::bitwiseXor_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-XOR to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<float>::bitwiseXor_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-XOR to operands of type (float, float).");
}

template<>
inline void Cleartext<double>::bitwiseXor_inplace(const AbstractValue &) {
  throw std::runtime_error("Cannot apply bitwise-XOR to operands of type (double, double).");
}

template<>
inline void Cleartext<int>::subtract_inplace(const AbstractValue &other) {
  // Subtraction needs special handling because in case of operands (Cleartext<T>, AbstractCiphertext), we cannot simply
  // swap both operands as subtraction is non-commutative. Instead, we need to transform the first operand into a
  // ciphertext and then perform the subtraction.
  if (auto otherCiphertext = dynamic_cast<const AbstractCiphertext *>(&other)) {
    auto cleartextData = getData();
    std::unique_ptr<AbstractCiphertext> thisCiphertext = otherCiphertext->getFactory().createCiphertext(cleartextData);
    thisCiphertext->subtractInplace(*otherCiphertext);
  } else {
    applyBinaryOperator(std::minus<>(), other);
  }
}

template<>
inline void Cleartext<float>::bitwiseNot_inplace() {
  throw std::runtime_error("Cannot apply bitwise-NOT to operand of type (float).");
}

template<>
inline void Cleartext<double>::bitwiseNot_inplace() {
  throw std::runtime_error("Cannot apply bitwise-NOT to operand of type (double).");
}

template<>
inline void Cleartext<std::string>::bitwiseNot_inplace() {
  throw std::runtime_error("Cannot apply bitwise-NOT to operand of type (std::string).");
}

template<>
inline void Cleartext<std::string>::logicalNot_inplace() {
  throw std::runtime_error("Cannot apply (logical) not to operand of type (std::string).");
}

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
