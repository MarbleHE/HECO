#ifndef AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_
#define AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_

#include <string>
#include <nlohmann/json.hpp>

class Dimension;
class AbstractLiteral;
class Operator;
class AbstractExpr;

template<typename T>
class Matrix;

using json = nlohmann::json;

/// A helper class that allows to access and use all Matrix<T> instances using a unified interface, without
/// needing exact knowledge about the specific template type T.
class AbstractMatrix {
 public:
  virtual AbstractMatrix *rotate(int rotationFactor, bool inPlace) = 0;

  virtual AbstractMatrix *transpose(bool inPlace) = 0;

  virtual Dimension &getDimensions() = 0;

  virtual std::string toString() = 0;

  [[nodiscard]] virtual bool isScalar() const = 0;

  [[nodiscard]] virtual json toJson() const = 0;

  virtual AbstractMatrix *applyBinaryOperator(AbstractMatrix *rhsOperand, Operator *os) = 0;

  virtual AbstractMatrix *applyUnaryOperatorComponentwise(Operator *os) = 0;

  virtual AbstractExpr *getElementAt(int row, int column) = 0;

  bool operator==(const AbstractMatrix &rhs) const;

  bool operator!=(const AbstractMatrix &rhs) const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_
