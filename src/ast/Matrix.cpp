#include "Matrix.h"

void throwUnknownOperatorException() {
  throw std::runtime_error("Unknown or unsupported operator/operands combination encountered! "
                           "Cannot determine correct Matrix operation.");
}

template<typename T>
AbstractMatrix *Matrix<T>::applyBinaryOperatorComponentwise(Matrix<T> *, Operator *) {
  throw std::runtime_error("applyOperatorComponentwise is unimplemented for type T: " + std::string(typeid(T).name()));
}

// template specialization for integers
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

// template specialization for floats
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

// template specialization for booleans
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

// template specialization for strings
template<>
AbstractMatrix *Matrix<std::string>::applyBinaryOperatorComponentwise(Matrix<std::string> *rhsOperand, Operator *os) {
  std::function<std::string(std::string, std::string)> operatorFunction;
  if (os->equals(ArithmeticOp::ADDITION)) {
    operatorFunction = [](const std::string &a, const std::string &b) -> std::string { return a + b; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(::applyComponentwise(
      dynamic_cast<Matrix<std::string> *>(this), dynamic_cast<Matrix<std::string> *>(rhsOperand), operatorFunction));
}

template<typename T>
AbstractMatrix *Matrix<T>::applyUnaryOperatorComponentwise(Operator *) {
  throw std::runtime_error("applyUnaryOpComponentwise is unimplemented for type T: " + std::string(typeid(T).name()));
}

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

template<typename T>
AbstractLiteral *Matrix<T>::getElementAt(int row, int column) {
  throw std::logic_error(
      "Value in matrix is of unknown type. Cannot determine associated AbstractLiteral subtype.");
}

template<>
AbstractLiteral *Matrix<int>::getElementAt(int row, int column) {
  checkMatrixIndexAccess(row, column);
  return new LiteralInt(values[row][column]);
}

template<>
AbstractLiteral *Matrix<float>::getElementAt(int row, int column) {
  checkMatrixIndexAccess(row, column);
  return new LiteralFloat(values[row][column]);
}

template<>
AbstractLiteral *Matrix<std::string>::getElementAt(int row, int column) {
  checkMatrixIndexAccess(row, column);
  return new LiteralString(values[row][column]);
}

template<>
AbstractLiteral *Matrix<bool>::getElementAt(int row, int column) {
  checkMatrixIndexAccess(row, column);
  return new LiteralBool(values[row][column]);
}
