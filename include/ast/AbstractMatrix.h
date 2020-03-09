#ifndef AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_
#define AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_

#include <string>
#include <nlohmann/json.hpp>

class Dimension;
class AbstractLiteral;
class Operator;

using json = nlohmann::json;

/// A helper class that allows to access and use all Matrix<T> specializations using a unified interface.
class AbstractMatrix {
 public:
  virtual AbstractMatrix *rotate(int rotationFactor, bool inPlace) = 0;

  virtual Dimension &getDimensions() = 0;

  virtual std::string toString() = 0;

  [[nodiscard]] virtual bool isScalar() const = 0;

  [[nodiscard]] virtual json toJson() const = 0;

  virtual AbstractLiteral *applyOperatorComponentwise(AbstractMatrix *rhsOperand, Operator *op) = 0;

  virtual AbstractLiteral *applyMatrixMultiplication(AbstractMatrix *rhsOperand) = 0;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_ABSTRACTMATRIX_H_
