#include "Matrix.h"

void throwUnknownOperatorException() {
  throw std::runtime_error("Unknown or unsupported operator/operands combination encountered! "
                           "Cannot determine correct Matrix operation.");
}

template<typename T>
AbstractMatrix *Matrix<T>::applyBinaryOperatorComponentwise(Matrix<T> *rhsOperand, Operator *os) {
  throw std::runtime_error("applyOperatorComponentwise is unimplemented for type T: " + std::string(typeid(T).name()));
}

// template specialization for integers
template<>
AbstractMatrix *Matrix<int>::applyBinaryOperatorComponentwise(Matrix<int> *rhsOperand, Operator *os) {
  std::function<int(int, int)> operatorFunction;
  if (os->equals(ArithmeticOp::addition)) {            // ==== Arithmetic Operators ===============================
    operatorFunction = [](int a, int b) -> int { return a + b; };
  } else if (os->equals(ArithmeticOp::subtraction)) {
    operatorFunction = [](int a, int b) -> int { return a - b; };
  } else if (os->equals(ArithmeticOp::division)) {
    operatorFunction = [](int a, int b) -> int { return a/b; };
  } else if (os->equals(ArithmeticOp::multiplication)) {
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
  } else if (os->equals(ArithmeticOp::modulo)) { // only for integers
    operatorFunction = [](int a, int b) -> int { return a%b; };
  } else if (os->equals(LogCompOp::logicalAnd)) {        // ==== Logical Operators ================================
    operatorFunction = [](int a, int b) -> int { return a && b; };
  } else if (os->equals(LogCompOp::logicalOr)) {
    operatorFunction = [](int a, int b) -> int { return a || b; };
  } else if (os->equals(LogCompOp::logicalXor)) {
    operatorFunction = [](int a, int b) -> int { return a ^ b; };
  } else if (os->equals(LogCompOp::smaller)) {           // ==== Comparison Operators =============================
    operatorFunction = [](int a, int b) -> int { return a < b; };
  } else if (os->equals(LogCompOp::smallerEqual)) {
    operatorFunction = [](int a, int b) -> int { return a <= b; };
  } else if (os->equals(LogCompOp::greater)) {
    operatorFunction = [](int a, int b) -> int { return a > b; };
  } else if (os->equals(LogCompOp::greaterEqual)) {
    operatorFunction = [](int a, int b) -> int { return a >= b; };
  } else if (os->equals(LogCompOp::equal)) {
    operatorFunction = [](int a, int b) -> int { return a==b; };
  } else if (os->equals(LogCompOp::unequal)) {
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
  if (os->equals(ArithmeticOp::addition)) {            // ==== Arithmetic Operators ===============================
    operatorFunction = [](float a, float b) -> float { return a + b; };
  } else if (os->equals(ArithmeticOp::subtraction)) {
    operatorFunction = [](float a, float b) -> float { return a - b; };
  } else if (os->equals(ArithmeticOp::division)) {
    operatorFunction = [](float a, float b) -> float { return a/b; };
  } else if (os->equals(ArithmeticOp::multiplication)) {
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
  } else if (os->equals(LogCompOp::logicalAnd)) {        // ==== Logical Operators ================================
    operatorFunction = [](float a, float b) -> float { return a && b; };
  } else if (os->equals(LogCompOp::logicalOr)) {
    operatorFunction = [](float a, float b) -> float { return a || b; };
  } else if (os->equals(LogCompOp::smaller)) {           // ==== Comparison Operators =============================
    operatorFunction = [](float a, float b) -> float { return a < b; };
  } else if (os->equals(LogCompOp::smallerEqual)) {
    operatorFunction = [](float a, float b) -> float { return a <= b; };
  } else if (os->equals(LogCompOp::greater)) {
    operatorFunction = [](float a, float b) -> float { return a > b; };
  } else if (os->equals(LogCompOp::greaterEqual)) {
    operatorFunction = [](float a, float b) -> float { return a >= b; };
  } else if (os->equals(LogCompOp::equal)) {
    operatorFunction = [](float a, float b) -> float { return a==b; };
  } else if (os->equals(LogCompOp::unequal)) {
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
  if (os->equals(ArithmeticOp::addition)) {            // ==== Arithmetic Operators ===============================
    operatorFunction = [](bool a, bool b) -> bool { return a + b; };
  } else if (os->equals(ArithmeticOp::subtraction)) {
    operatorFunction = [](bool a, bool b) -> bool { return a - b; };
  } else if (os->equals(ArithmeticOp::multiplication)) {
    // For scalar multiplication: one is vector, other is scalar.
    // If both operands are matrices, then applyMatrixMultiplication should have been called instead.
    operatorFunction = [](bool a, bool b) -> bool { return a*b; };
  } else if (os->equals(LogCompOp::logicalAnd)) {        // ==== Logical Operators ================================
    operatorFunction = [](bool a, bool b) -> bool { return a && b; };
  } else if (os->equals(LogCompOp::logicalOr)) {
    operatorFunction = [](bool a, bool b) -> bool { return a || b; };
  } else if (os->equals(LogCompOp::logicalXor)) {
    operatorFunction = [](int a, int b) -> int { return a ^ b; };
  } else if (os->equals(LogCompOp::equal)) {
    operatorFunction = [](bool a, bool b) -> bool { return a==b; };    // ==== Comparison Operators ====================
  } else if (os->equals(LogCompOp::unequal)) {
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
  if (os->equals(ArithmeticOp::addition)) {
    operatorFunction = [](std::string a, std::string b) -> std::string { return a + b; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(::applyComponentwise(
      dynamic_cast<Matrix<std::string> *>(this), dynamic_cast<Matrix<std::string> *>(rhsOperand), operatorFunction));
}

template<typename T>
AbstractMatrix *Matrix<T>::applyUnaryOperatorComponentwise(Operator *os) {
  throw std::runtime_error("applyUnaryOpComponentwise is unimplemented for type T: " + std::string(typeid(T).name()));
}

template<>
AbstractMatrix *Matrix<int>::applyUnaryOperatorComponentwise(Operator *os) {
  std::function<int(int)> operatorFunction;
  if (os->equals(UnaryOp::negation)) {
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
  if (os->equals(UnaryOp::negation)) {
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
  if (os->equals(UnaryOp::negation)) {
    operatorFunction = [](bool elem) -> bool { return !elem; };
  } else {
    throwUnknownOperatorException();
  }
  return reinterpret_cast<AbstractMatrix *>(
      ::applyOnEachElement(dynamic_cast<Matrix<bool> *>(this), operatorFunction));
}
