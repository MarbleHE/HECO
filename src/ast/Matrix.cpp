#include "ast_opt/ast/Matrix.h"

void throwUnknownOperatorException() {
  throw std::runtime_error("Unknown or unsupported operator/operands combination encountered! "
                           "Cannot determine correct Matrix operation.");
}

template<>
bool Matrix<AbstractExpr *>::operator==(const Matrix &rhs) const {
  if (dim!=rhs.dim) return false;

  for (int i = 0; i < values.size(); ++i) {
    for (int j = 0; j < values[i].size(); ++j) {
      if (values[i][j]==nullptr && rhs.values[i][j]==nullptr) continue;
      if ((values[i][j]==nullptr && rhs.values[i][j]!=nullptr)
          || (values[i][j]!=nullptr && rhs.values[i][j]==nullptr)
          || (!values[i][j]->isEqual(rhs.values[i][j]))) {
        return false;
      }
    }
  }
  return true;
}

// ===== applyBinaryOperatorComponentwise ==========
// - Matrix<int>
// - Matrix<float>
// - Matrix<bool>
// - Matrix<std::string>

template<>
AbstractMatrix *Matrix<int>::applyBinaryOperatorComponentwise(Matrix<int> *rhsOperand, Operator *os) {
  std::function<int(int, int)> operatorFunction;
  if (os->equals(ArithmeticOp::ADDITION)) {            // ==== Arithmetic Operators ===============================
    operatorFunction = [](int a, int b) -> int { return a + b; };
  } else if (os->equals(ArithmeticOp::SUBTRACTION)) {
    operatorFunction = [](int a, int b) -> int { return a - b; };
  } else if (os->equals(ArithmeticOp::DIVISION)) {
    operatorFunction = [](int a, int b) -> int { return a/b; };
  } else if (os->equals(ArithmeticOp::MULTIPLICATION)) {
    // if exactly one of both is a scalar -> use matrix * scalar multiplication
    if (this->isScalar() ^ rhsOperand->isScalar()) {
      // For scalar multiplication: one is vector, other is scalar.
      // If both operands are matrices, then applyMatrixMultiplication should have been called instead.
      operatorFunction = [](int a, int b) -> int { return a*b; };
    } else {
      // otherwise both are matrices -> do matrix multiplication
      return reinterpret_cast<AbstractMatrix *>(::applyMatrixMultiplication(
          dynamic_cast<Matrix<int> *>(this), dynamic_cast<Matrix<int> *>(rhsOperand)));
    }
  } else if (os->equals(ArithmeticOp::MODULO)) { // only for integers
    operatorFunction = [](int a, int b) -> int { return a%b; };
  } else if (os->equals(LogCompOp::LOGICAL_AND)) {        // ==== Logical Operators ===============================
    operatorFunction = [](int a, int b) -> int { return a && b; };
  } else if (os->equals(LogCompOp::LOGICAL_OR)) {
    operatorFunction = [](int a, int b) -> int { return a || b; };
  } else if (os->equals(LogCompOp::SMALLER)) {           // ==== Comparison Operators =============================
    operatorFunction = [](int a, int b) -> int { return a < b; };
  } else if (os->equals(LogCompOp::SMALLER_EQUAL)) {
    operatorFunction = [](int a, int b) -> int { return a <= b; };
  } else if (os->equals(LogCompOp::GREATER)) {
    operatorFunction = [](int a, int b) -> int { return a > b; };
  } else if (os->equals(LogCompOp::GREATER_EQUAL)) {
    operatorFunction = [](int a, int b) -> int { return a >= b; };
  } else if (os->equals(LogCompOp::EQUAL)) {
    operatorFunction = [](int a, int b) -> int { return a==b; };
  } else if (os->equals(LogCompOp::UNEQUAL)) {
    operatorFunction = [](int a, int b) -> int { return a!=b; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(::applyComponentwise(
      dynamic_cast<Matrix<int> *>(this), dynamic_cast<Matrix<int> *>(rhsOperand), operatorFunction));
}

template<>
AbstractMatrix *Matrix<float>::applyBinaryOperatorComponentwise(Matrix<float> *rhsOperand, Operator *os) {
  std::function<float(float, float)> operatorFunction;
  if (os->equals(ArithmeticOp::ADDITION)) {            // ==== Arithmetic Operators ===============================
    operatorFunction = [](float a, float b) -> float { return a + b; };
  } else if (os->equals(ArithmeticOp::SUBTRACTION)) {
    operatorFunction = [](float a, float b) -> float { return a - b; };
  } else if (os->equals(ArithmeticOp::DIVISION)) {
    operatorFunction = [](float a, float b) -> float { return a/b; };
  } else if (os->equals(ArithmeticOp::MULTIPLICATION)) {
    // if exactly one of both is a scalar -> use matrix * scalar multiplication
    if (this->isScalar() ^ rhsOperand->isScalar()) {
      // For scalar multiplication: one is vector, other is scalar.
      // If both operands are matrices, then applyMatrixMultiplication should have been called instead.
      operatorFunction = [](float a, float b) -> float { return a*b; };
    } else {
      // otherwise both are matrices -> do matrix multiplication
      return reinterpret_cast<AbstractMatrix *>(::applyMatrixMultiplication(
          dynamic_cast<Matrix<float> *>(this), dynamic_cast<Matrix<float> *>(rhsOperand)));
    }
  } else if (os->equals(LogCompOp::SMALLER)) {           // ==== Comparison Operators =============================
    operatorFunction = [](float a, float b) -> float { return a < b; };
  } else if (os->equals(LogCompOp::SMALLER_EQUAL)) {
    operatorFunction = [](float a, float b) -> float { return a <= b; };
  } else if (os->equals(LogCompOp::GREATER)) {
    operatorFunction = [](float a, float b) -> float { return a > b; };
  } else if (os->equals(LogCompOp::GREATER_EQUAL)) {
    operatorFunction = [](float a, float b) -> float { return a >= b; };
  } else if (os->equals(LogCompOp::EQUAL)) {
    operatorFunction = [](float a, float b) -> float { return a==b; };
  } else if (os->equals(LogCompOp::UNEQUAL)) {
    operatorFunction = [](float a, float b) -> float { return a!=b; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(::applyComponentwise(
      dynamic_cast<Matrix<float> *>(this), dynamic_cast<Matrix<float> *>(rhsOperand), operatorFunction));
}

template<>
AbstractMatrix *Matrix<bool>::applyBinaryOperatorComponentwise(Matrix<bool> *rhsOperand, Operator *os) {
  std::function<bool(bool, bool)> operatorFunction;
  if (os->equals(ArithmeticOp::ADDITION)) {            // ==== Arithmetic Operators ===============================
    operatorFunction = [](bool a, bool b) -> bool { return a + b; };
  } else if (os->equals(ArithmeticOp::SUBTRACTION)) {
    operatorFunction = [](bool a, bool b) -> bool { return a - b; };
  } else if (os->equals(ArithmeticOp::MULTIPLICATION)) {
    // For scalar multiplication: one is vector, other is scalar.
    // If both operands are matrices, then applyMatrixMultiplication should have been called instead.
    operatorFunction = [](bool a, bool b) -> bool { return a*b; };
  } else if (os->equals(LogCompOp::LOGICAL_AND)) {        // ==== Logical Operators ===============================
    operatorFunction = [](bool a, bool b) -> bool { return a && b; };
  } else if (os->equals(LogCompOp::LOGICAL_OR)) {
    operatorFunction = [](bool a, bool b) -> bool { return a || b; };
  } else if (os->equals(LogCompOp::LOGICAL_XOR)) {
    operatorFunction = [](bool a, bool b) -> int { return a ^ b; };
  } else if (os->equals(LogCompOp::EQUAL)) {
    operatorFunction = [](bool a, bool b) -> bool { return a==b; };    // ==== Comparison Operators ====================
  } else if (os->equals(LogCompOp::UNEQUAL)) {
    operatorFunction = [](bool a, bool b) -> bool { return a!=b; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(::applyComponentwise(
      dynamic_cast<Matrix<bool> *>(this), dynamic_cast<Matrix<bool> *>(rhsOperand), operatorFunction));
}

template<>
AbstractMatrix *Matrix<std::string>::applyBinaryOperatorComponentwise(Matrix<std::string> *rhsOperand, Operator *os) {
  std::function < std::string(std::string, std::string) > operatorFunction;
  if (os->equals(ArithmeticOp::ADDITION)) {
    operatorFunction = [](const std::string &a, const std::string &b) -> std::string { return a + b; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(::applyComponentwise(
      dynamic_cast<Matrix<std::string> *>(this), dynamic_cast<Matrix<std::string> *>(rhsOperand), operatorFunction));
}

// ===== applyUnaryOperatorComponentwise ==========
// - Matrix<int>
// - Matrix<float>
// - Matrix<bool>

template<>
AbstractMatrix *Matrix<int>::applyUnaryOperatorComponentwise(Operator *os) {
  std::function<int(int)> operatorFunction;
  if (os->equals(UnaryOp::NEGATION)) {
    operatorFunction = [](int elem) -> int { return -elem; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(
      ::applyOnEachElement(dynamic_cast<Matrix<int> *>(this), operatorFunction));
}

template<>
AbstractMatrix *Matrix<float>::applyUnaryOperatorComponentwise(Operator *os) {
  std::function<float(float)> operatorFunction;
  if (os->equals(UnaryOp::NEGATION)) {
    operatorFunction = [](float elem) -> float { return -elem; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(
      ::applyOnEachElement(dynamic_cast<Matrix<float> *>(this), operatorFunction));
}

template<>
AbstractMatrix *Matrix<bool>::applyUnaryOperatorComponentwise(Operator *os) {
  std::function<bool(bool)> operatorFunction;
  if (os->equals(UnaryOp::NEGATION)) {
    operatorFunction = [](bool elem) -> bool { return !elem; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(
      ::applyOnEachElement(dynamic_cast<Matrix<bool> *>(this), operatorFunction));
}

// ===== getElementAt ==========
// - Matrix<int>
// - Matrix<float>
// - Matrix<bool>
// - Matrix<std::string>
// - Matrix<AbstractExpr*>

template<>
AbstractExpr *Matrix<int>::getElementAt(int row, int column) {
  boundCheckMatrixAccess(row, column);
  return new LiteralInt(values[row][column]);
}

template<>
AbstractExpr *Matrix<float>::getElementAt(int row, int column) {
  boundCheckMatrixAccess(row, column);
  return new LiteralFloat(values[row][column]);
}

template<>
AbstractExpr *Matrix<bool>::getElementAt(int row, int column) {
  boundCheckMatrixAccess(row, column);
  return new LiteralBool(values[row][column]);
}

template<>
AbstractExpr *Matrix<std::string>::getElementAt(int row, int column) {
  boundCheckMatrixAccess(row, column);
  return new LiteralString(values[row][column]);
}

template<>
AbstractExpr *Matrix<AbstractExpr *>::getElementAt(int row, int column) {
  boundCheckMatrixAccess(row, column);
  return values[row][column];
}

// ===== Matrix Constructor (additional constructor) ==========
// - Matrix<AbstractExpr*>

template<>
Matrix<AbstractExpr *>::Matrix(std::vector<std::vector<AbstractExpr *>> inputMatrix)  /* NOLINT intentionally not explicit */
    : values(std::move(inputMatrix)), dim(Dimension(values.size(), values.size()==0 ? 0 : values.at(0).size())) {
  // In a Matrix<AbstractExpr*> it is needed that we use the parent-child relationship by attaching each of the matrix
  // elements as a child to the Matrix object. This is needed, for example, in the CompileTimeExpressionSimplifier where
  // we replace nodes that can be evaluated at compile-time (e.g., variables by their known value).
  int elementsPerRow = values.size()==0 ? 0 : values.at(0).size();
  std::vector<AbstractNode *> childrenToBeAdded;
  for (auto &matrixRows : values) {
    // check that matrix has the same number of elements in each row
    if (matrixRows.size()!=elementsPerRow) {
      throw std::invalid_argument("Vector rows must all have the same number of elements!");
    }
    for (auto &element : matrixRows) {
      // check that matrix elements are all one-dimensional
      auto lit = dynamic_cast<AbstractLiteral *>(element);
      if (lit!=nullptr && !lit->getMatrix()->getDimensions().equals(1, 1)) {
        throw std::logic_error("Cannot create a matrix where elements are not one-dimensional!");
      }
      childrenToBeAdded.push_back(element);
    }
  }
  removeChildren();
  addChildren(childrenToBeAdded, true);
}

// ===== toJson ==========
// - Matrix<AbstractExpr*>
// - Matrix<bool>
// - Matrix<int>
// - Matrix<float>
// - Matrix<double>
// - Matrix<std::string>

template<>
[[nodiscard]] json Matrix<AbstractExpr *>::toJson() const {
  json jsonOutput = json::array();
  for (const auto &matrixRow : values) {
    json jsonMatrixRow = json::array();
    for (auto matrixElement : matrixRow) {
      jsonMatrixRow.push_back(matrixElement->toJson());
    }
    jsonOutput.push_back(jsonMatrixRow);
  }
  return jsonOutput;
}
template<>
json Matrix<bool>::toJson() const {
  // Return the scalar value if this is a (1,1) scalar matrix
  if (isScalar()) return json(getScalarValue());
  // If this is a matrix of dimension (M,N), return an array of arrays like
  //   [ [a00, a01, a02], [a10, a11, a12], ..., [aN0, aN1, ..., aMM] ],
  // where each inner array represents a matrix row.
  json arrayOfArrays = json::array();
  for (int i = 0; i < values.size(); ++i) {
    arrayOfArrays.push_back(json(values[i]));
  }
  return arrayOfArrays;
}
template<>
json Matrix<int>::toJson() const {
  // Return the scalar value if this is a (1,1) scalar matrix
  if (isScalar()) return json(getScalarValue());
  // If this is a matrix of dimension (M,N), return an array of arrays like
  //   [ [a00, a01, a02], [a10, a11, a12], ..., [aN0, aN1, ..., aMM] ],
  // where each inner array represents a matrix row.
  json arrayOfArrays = json::array();
  for (int i = 0; i < values.size(); ++i) {
    arrayOfArrays.push_back(json(values[i]));
  }
  return arrayOfArrays;
}
template<>
json Matrix<float>::toJson() const {
  // Return the scalar value if this is a (1,1) scalar matrix
  if (isScalar()) return json(getScalarValue());
  // If this is a matrix of dimension (M,N), return an array of arrays like
  //   [ [a00, a01, a02], [a10, a11, a12], ..., [aN0, aN1, ..., aMM] ],
  // where each inner array represents a matrix row.
  json arrayOfArrays = json::array();
  for (int i = 0; i < values.size(); ++i) {
    arrayOfArrays.push_back(json(values[i]));
  }
  return arrayOfArrays;
}

template<>
json Matrix<double>::toJson() const {
  // Return the scalar value if this is a (1,1) scalar matrix
  if (isScalar()) return json(getScalarValue());
  // If this is a matrix of dimension (M,N), return an array of arrays like
  //   [ [a00, a01, a02], [a10, a11, a12], ..., [aN0, aN1, ..., aMM] ],
  // where each inner array represents a matrix row.
  json arrayOfArrays = json::array();
  for (int i = 0; i < values.size(); ++i) {
    arrayOfArrays.push_back(json(values[i]));
  }
  return arrayOfArrays;
}

template<>
json Matrix<std::string>::toJson() const {
  // Return the scalar value if this is a (1,1) scalar matrix
  if (isScalar()) return json(getScalarValue());
  // If this is a matrix of dimension (M,N), return an array of arrays like
  //   [ [a00, a01, a02], [a10, a11, a12], ..., [aN0, aN1, ..., aMM] ],
  // where each inner array represents a matrix row.
  json arrayOfArrays = json::array();
  for (int i = 0; i < values.size(); ++i) {
    arrayOfArrays.push_back(json(values[i]));
  }
  return arrayOfArrays;
}

// ===== clone ==========
// - Matrix<AbstractExpr*>

template<>
Matrix<AbstractExpr *> *Matrix<AbstractExpr *>::clone(bool keepOriginalUniqueNodeId) {
  std::vector<std::vector<AbstractExpr *>> clonedMatrix(dim.numRows, std::vector<AbstractExpr *>(dim.numColumns));
  // we need to clone each AbstractExpr contained in this matrix
  for (int i = 0; i < values.size(); ++i) {
    for (int j = 0; j < values[i].size(); ++j) {
      clonedMatrix[i][j] = values[i][j]->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>();
    }
  }
  return new Matrix<AbstractExpr *>(clonedMatrix);
}

// ===== addElementToStringStream ==========
// - Matrix<AbstractExpr*>

template<>
void Matrix<AbstractExpr *>::addElementToStringStream(AbstractExpr *elem, std::stringstream &s) {
  // Although wrongly indicated by CLion, this method is actually used, see Matrix<T>::toString().
  s << *elem;
}

// ===== setElementAt  ==========
// - Matrix<AbstractExpr*>
// - Matrix<int>
// - Matrix<float>
// - Matrix<bool>
// - Matrix<std::string>


template<>
void Matrix<AbstractExpr *>::setElementAt(int row, int column, AbstractExpr *element) {
  checkBoundsAndResizeMatrix(row, column);
  values[row][column] = element;
  getDimensions().update(values.size(), values.at(0).size());
}

template<>
void Matrix<int>::setElementAt(int row, int column, AbstractExpr *element) {
  if (auto elementAsLiteral = dynamic_cast<LiteralInt *>(element)) {
    checkBoundsAndResizeMatrix(row, column);
    values[row][column] = elementAsLiteral->getValue();
    getDimensions().update(values.size(), values.at(0).size());
  } else {
    throw std::runtime_error("Unexpected element given that cannot be added to Matrix<int>.");
  }
}

template<>
void Matrix<float>::setElementAt(int row, int column, AbstractExpr *element) {
  if (auto elementAsLiteral = dynamic_cast<LiteralFloat *>(element)) {
    checkBoundsAndResizeMatrix(row, column);
    values[row][column] = elementAsLiteral->getValue();
    getDimensions().update(values.size(), values.at(0).size());
  } else {
    throw std::runtime_error("Unexpected element given that cannot be added to Matrix<float>.");
  }
}

template<>
void Matrix<bool>::setElementAt(int row, int column, AbstractExpr *element) {
  if (auto elementAsLiteral = dynamic_cast<LiteralBool *>(element)) {
    checkBoundsAndResizeMatrix(row, column);
    values[row][column] = elementAsLiteral->getValue();
    getDimensions().update(values.size(), values.at(0).size());
  } else {
    throw std::runtime_error("Unexpected element given that cannot be added to Matrix<bool>.");
  }
}

template<>
void Matrix<std::string>::setElementAt(int row, int column, AbstractExpr *element) {
  if (auto elementAsLiteral = dynamic_cast<LiteralString *>(element)) {
    checkBoundsAndResizeMatrix(row, column);
    values[row][column] = elementAsLiteral->getValue();
    getDimensions().update(values.size(), values.at(0).size());
  } else {
    throw std::runtime_error("Unexpected element given that cannot be added to Matrix<std::string>.");
  }
}

// ===== replaceChild ==========
// - Matrix<AbstractExpr*>

template<>
void Matrix<AbstractExpr *>::replaceChild(AbstractNode *originalChild, AbstractNode *newChildToBeAdded) {
  for (int i = 0; i < values.size(); ++i) {
    for (int j = 0; j < values[i].size(); ++j) {
      if (values[i][j]==originalChild) {
        setElementAt(i, j, dynamic_cast<AbstractExpr *>(newChildToBeAdded));
        break;
      }
    }
  }
  AbstractNode::replaceChild(originalChild, newChildToBeAdded);
}

// ===== containsAbstractExprs ==========
// - Matrix<AbstractExpr*>

template<>
bool Matrix<AbstractExpr *>::containsAbstractExprs() {
  return true;
}