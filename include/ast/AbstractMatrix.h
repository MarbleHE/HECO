#ifndef AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_
#define AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_

#include <string>
#include <nlohmann/json.hpp>

class Dimension;
class AbstractLiteral;
class Operator;

template<typename T>
class Matrix;

using json = nlohmann::json;

/// A helper class that allows to access and use all Matrix<T> specializations using a unified interface.
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

  virtual AbstractLiteral *getElementAt(int row, int column) = 0;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_
