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
        throw std::runtime_error("");
      }
    }
  }

  explicit Cleartext(std::vector<T> values) {
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

  void applyOperator(std::function<T(T, T)> f, AbstractValue &other) {
    if (auto otherCleartxt = dynamic_cast<Cleartext<T> *>(&other)) {
      std::transform(data.begin(), data.end(), otherCleartxt->data.begin(), data.begin(), f);
    } else {
      throw std::runtime_error(
          "Given operation can only be applied on (Cleartext<T>, Cleartext<T>). "
          "This could happen, for example, if an operation is called on (Cleartext<T>, AbstractCiphertext) "
          "but the operation is unsupported in FHE.");
    }
  }

  void add(AbstractValue &other) override {
    applyOperator(std::plus<T>(), other);
  }

  void subtract(AbstractValue &other) override {
    applyOperator(std::minus<T>(), other);
  }

  void multiply(AbstractValue &other) override {
    applyOperator(std::multiplies<T>(), other);
  }

  void divide(AbstractValue &other) override {
    applyOperator(std::divides<T>(), other);
  }

  void modulo(AbstractValue &other) override {
    applyOperator(std::modulus<T>(), other);
  }

  void logicalAnd(AbstractValue &other) override {
    applyOperator(std::logical_and<T>(), other);
  }

  void logicalOr(AbstractValue &other) override {
    applyOperator(std::logical_or<T>(), other);
  }

  void logicalLess(AbstractValue &other) override {
    applyOperator(std::less<T>(), other);
  }

  void logicalLessEqual(AbstractValue &other) override {
    applyOperator(std::less_equal<T>(), other);
  }

  void logicalGreater(AbstractValue &other) override {
    applyOperator(std::greater<T>(), other);
  }

  void logicalGreaterEqual(AbstractValue &other) override {
    applyOperator(std::greater_equal<T>(), other);
  }

  void logicalEqual(AbstractValue &other) override {
    applyOperator(std::equal_to<T>(), other);
  }

  void logicalNotEqual(AbstractValue &other) override {
    applyOperator(std::not_equal_to<T>(), other);
  }

  void logicalBitwiseAnd(AbstractValue &other) override {
    applyOperator(std::bit_and<T>(), other);
  }

  void logicalBitwiseXor(AbstractValue &other) override {
    applyOperator(std::bit_xor<T>(), other);
  }

  void logicalBitwiseOr(AbstractValue &other) override {
    applyOperator(std::bit_or<T>(), other);
  }
};


template<typename>
struct isCleartext : public std::false_type {};

template<typename T>
struct isCleartext<Cleartext<T>> : public std::true_type {};

// partial specializations of Cleartext<T> member functions

template<>
inline void Cleartext<std::string>::subtract(AbstractValue &other) {
  throw std::runtime_error("Cannot apply subtraction to operands of type (std::string, std:string).");
}

template<>
inline void Cleartext<std::string>::multiply(AbstractValue &other) {
  throw std::runtime_error("Cannot apply multiplication to operands of type (std::string, std:string).");
}

template<>
inline void Cleartext<std::string>::divide(AbstractValue &other) {
  throw std::runtime_error("Cannot division to operands of type (std::string, std:string).");
}

template<>
inline void Cleartext<float>::modulo(AbstractValue &other) {
  throw std::runtime_error("Cannot apply modulo to operands of type (float, float).");
}

template<>
inline void Cleartext<double>::modulo(AbstractValue &other) {
  throw std::runtime_error("Cannot apply modulo to operands of type (double, double).");
}

template<>
inline void Cleartext<std::string>::modulo(AbstractValue &other) {
  throw std::runtime_error("Cannot apply modulo to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalAnd(AbstractValue &other) {
  throw std::runtime_error("Cannot apply AND to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalOr(AbstractValue &other) {
  throw std::runtime_error("Cannot apply OR to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalLess(AbstractValue &other) {
  throw std::runtime_error("Cannot apply less to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalLessEqual(AbstractValue &other) {
  throw std::runtime_error("Cannot apply less-equal to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalGreater(AbstractValue &other) {
  throw std::runtime_error("Cannot apply greater to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalGreaterEqual(AbstractValue &other) {
  throw std::runtime_error("Cannot apply greater-equal to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalEqual(AbstractValue &other) {
  throw std::runtime_error("Cannot apply equal to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalNotEqual(AbstractValue &other) {
  throw std::runtime_error("Cannot apply not-equal to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<std::string>::logicalBitwiseAnd(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-AND to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<float>::logicalBitwiseAnd(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-AND to operands of type (float, float).");
}

template<>
inline void Cleartext<double>::logicalBitwiseAnd(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-AND to operands of type (double, double).");
}

template<>
inline void Cleartext<std::string>::logicalBitwiseOr(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-OR to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<float>::logicalBitwiseOr(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-OR to operands of type (float, float).");
}

template<>
inline void Cleartext<double>::logicalBitwiseOr(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-OR to operands of type (double, double).");
}

template<>
inline void Cleartext<std::string>::logicalBitwiseXor(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-XOR to operands of type (std::string, std::string).");
}

template<>
inline void Cleartext<float>::logicalBitwiseXor(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-XOR to operands of type (float, float).");
}

template<>
inline void Cleartext<double>::logicalBitwiseXor(AbstractValue &other) {
  throw std::runtime_error("Cannot apply bitwise-XOR to operands of type (double, double).");
}

template<>
inline void Cleartext<int>::subtract(AbstractValue &other) {
  if (auto otherCiphertext = dynamic_cast<AbstractCiphertext *>(&other)) {
    auto cleartextData = getData();
    std::unique_ptr<AbstractCiphertext> thisCiphertext = otherCiphertext->getFactory().createCiphertext(cleartextData);
    thisCiphertext->subtractInplace(*otherCiphertext);
  } else {
    applyOperator(std::minus<>(), other);
  }
}

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
