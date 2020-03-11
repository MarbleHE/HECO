#ifndef AST_OPTIMIZER_INCLUDE_AST_MATRIX_H_
#define AST_OPTIMIZER_INCLUDE_AST_MATRIX_H_

#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "AbstractMatrix.h"
#include "Dimension.h"
#include "Operator.h"

using json = nlohmann::json;

template<typename T>
static Matrix<T> *applyComponentwise(Matrix<T> *A, Matrix<T> *B, std::function<T(T, T)> f) {
  // check that A and B have valid dimensions:
  // - if A and B have same dimensions -> apply f componentwise
  // - if either A or B is a scalar -> apply scalar on each component of matrix/vector
  bool isScalarA = A->isScalar();
  bool isScalarB = B->isScalar();
  if (A->getDimensions()!=B->getDimensions() && !(isScalarA ^ isScalarB)) {
    throw std::logic_error("Matrices A and B must have the same dimensions to apply an operator componentwise, or "
                           "either one of A and B must be a scalar to apply the scalar product.");
  } else if (isScalarA && !isScalarB) {
    // expand A to be a (#rowsB, #columnsB)-matrix filled with the value of the scalar A
    A->expandAndFillMatrix(B->getDimensions(), A->getScalarValue());
  } else if (isScalarB && !isScalarA) {
    // expand B to be a (#rowsA, #columnsA)-matrix filled with the value of the scalar B
    B->expandAndFillMatrix(A->getDimensions(), B->getScalarValue());
  }

  // Compute new matrix by applying binary function f componentwise on both matrices.
  // For example, for matrices A,B with each dim (M, N), the resulting matrix would look like:
  //    f(a_00, b_00)  f(a_01, b_01)  ...  f(a_0N, b_0N)
  //    f(a_10, b_10)      ...        ...      ...
  //         ...
  //    f(a_M0, b_M0)      ...        ...  f(a_MN, b_MN)
  std::vector<std::vector<T>> result;
  for (int i = 0; i < A->values.size(); ++i) {
    result.push_back(std::vector<T>());
    for (int j = 0; j < A->values[0].size(); ++j) {
      result[i].push_back(f((*A)(i, j), (*B)(i, j)));
    }
  }
  return new Matrix<T>(result);
}

template<typename T>
static Matrix<T> *applyOnEachElement(Matrix<T> *A, std::function<T(T)> f) {
  // Compute new matrix by applying unary function f on each element of A, e.g., for (M, N)-matrix A:
  //    f(a_00)    f(a_01)    ...    f(a_0N)
  //    f(a_10)      ...      ...      ...
  //     ...
  //    f(a_M0)      ...      ...    f(a_MN)
  std::vector<std::vector<T>> result;
  for (int i = 0; i < A->values.size(); ++i) {
    result.push_back(std::vector<T>());
    for (int j = 0; j < A->values[0].size(); ++j) {
      result[i].push_back(f((*A)(i, j)));
    }
  }
  return new Matrix<T>(result);
}

template<typename T>
static Matrix<T> *applyMatrixMultiplication(Matrix<T> *matrixA, Matrix<T> *matrixB) {
  // check that #column of matrixA equals #rows of matrixB
  if (!matrixA->getDimensions().equals(-1, matrixB->getDimensions().numRows)) {
    std::stringstream ss;
    ss << "Dimension mismatch: ";
    ss << "matrixA has dimensions " << matrixA->getDimensions() << " and matrixB has dimensions "
       << matrixB->getDimensions() << ". ";
    ss << "To apply matrix multiplication, #columns of matrixA must be equal to the #rows of matrixB." << std::endl;
    throw std::logic_error(ss.str());
  }

  // define result matrix having dimension (#rowsA, #columnsB)
  std::vector<std::vector<T>> result(matrixA->getDimensions().numRows,
                                     std::vector<T>(matrixB->getDimensions().numColumns));

  // matrixA has dim (M,N), matrixB has dim (N,P)
  int dimM = matrixA->getDimensions().numRows;
  int dimN = matrixA->getDimensions().numColumns;
  int dimP = matrixB->getDimensions().numColumns;
  T sum = 0;
  // Very inefficient: O(n^3)! Replace it, for example, by Strassen's multiplication algorithm.
  // For matrices A with dim (M,N) and B with dim (N,P) the following algorithm computes C with dim (M,P) where
  //  c_ij = sum_{k=1}^{n} a_ik * b_kj.
  // See, for example, Wikipedia for details: https://en.wikipedia.org/wiki/Matrix_multiplication.
  for (int k = 0; k < dimN; ++k) {
    for (int i = 0; i < dimM; ++i) {
      for (int j = 0; j < dimP; ++j) {
        result[i][j] += (*matrixA)(i, k)*(*matrixB)(k, j);
      }
    }
  }
  return new Matrix<T>(result);
}

template<typename T>
class Matrix : public AbstractMatrix {
 private:
  /// Computes the iterator n_first that is required by the std::rotate method and that determines the rotation factor
  /// (i.e., the number of elements and the direction of the rotation).
  /// \tparam Iterator An std::vector iterator.
  /// \param itBegin The begin iterator of the std::vector (i.e., std::vector.begin()).
  /// \param vectorSize The vector's size (i.e., std::vector.size()).
  /// \param rotationFactor The number of elements to be rotated. For example, rotationFactor=1 rotates all elements by
  ///        one to the right. A negative rotationFactor would instead rotate all elements to the left.
  /// \return An iterator that points to the element that should appear at the beginning of the rotated range.
  ///         See parameter n_first in std::rotate.
  template<typename Iterator>
  Iterator computeRotationTarget(Iterator itBegin, size_t vectorSize, int rotationFactor) {
    return (rotationFactor > 0) ? (itBegin + vectorSize - rotationFactor) : (itBegin - rotationFactor);
  }

 public:
  /// a matrix of row vectors
  std::vector<std::vector<T>> values;

  /// the dimension of the matrix
  Dimension dim;

  /// Creates a new matrix with the elements provided in inputMatrix.
  /// \param inputMatrix
  Matrix(std::vector<std::vector<T>> inputMatrix)  /* NOLINT intentionally not explicit */
      : values(std::move(inputMatrix)), dim(Dimension(values.size(), values.at(0).size())) {
    int elementsPerRow = values.at(0).size();
    for (auto const &rowVector : values) {
      if (rowVector.size()!=elementsPerRow) {
        throw std::invalid_argument("Vector rows must all have the same number of elements!");
      }
    }
  }

  /// Creates a new (1,1)-matrix consisting of a single value.
  /// \param scalarValue The value to be used to create this new one-element "scalar" matrix.
  Matrix(T scalarValue) : dim({1, 1}) {  /* NOLINT intentionally not explicit */
    values = {{scalarValue}};
  }

  /// Creates a new matrix by copying the values of another, existing matrix other.
  /// \param other The matrix to be copied to create the new matrix.
  Matrix(Matrix<T> &other) : dim(Dimension(other.getDimensions().numRows, other.getDimensions().numColumns)) {
    values = other.values;
  }

  /// Checks whether this matrix is a scalar, i.e., has dimension (1,1).
  /// \return True if this matrix is a scalar, otherwise False.
  [[nodiscard]] bool isScalar() const override {
    return dim.equals(1, 1);
  }

  /// Returns the only element of the (scalar) matrix if this matrix consists of a single element only, otherwise throws
  /// an exception to notify the user that this is not a scalar.
  /// \throws std::logic_error if this matrix has more than one value.
  /// \return The single element of the matrix.
  T getScalarValue() const {
    if (isScalar()) return values[0][0];
    throw std::logic_error("getScalarValue() not allowed on non-scalars!");
  }

  bool operator==(const Matrix &rhs) const {
    return values==rhs.values && dim==rhs.dim;
  }

  bool operator!=(const Matrix &rhs) const {
    return !(rhs==*this);
  }

  /// Creates a JSON representation of this matrix.
  /// \return The JSON representation of this matrix.
  [[nodiscard]] json toJson() const override {
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

  /// Returns a reference to the element at index specified by the given (row, column) pair.
  /// \param row The row number of the element to be returned a reference to.
  /// \param column The column number of the element to be returned a reference to.
  /// \return A reference to the element at position (row, column).
  // Returning std::vector<T>::reference is required here.
  // Credits to Mike Seymour from stackoverflow.com (https://stackoverflow.com/a/25770060/3017719).
  typename std::vector<T>::reference operator()(int row, int column) {
    if (!dim.isValidAccess(row, column)) {
      std::stringstream ss;
      ss << "Cannot access " << Dimension(row, column) << " because vector has dimensions " << getDimensions() << ".";
      throw std::invalid_argument(ss.str());
    }
    return values[row][column];
  }

  /// Returns a copy of the element at index specified by the given (row, column) pair.
  /// \param row The row number of the element to be returned as a copy.
  /// \param column The column number of the element to be returned as a copy.
  /// \return A copy of the element at position (row, column).
  T getElement(int row, int column) {
    return values[row][column];
  }

  /// Returns the dimension object that indicates the matrix dimensions.
  /// \return A reference to the dimension object associated to this matrix.
  Dimension &getDimensions() override {
    return dim;
  }

  /// Overwrites the matrix's values by the new values given.
  /// \param newValues The values to be used to overwrite the matrix's existing values.
  void setValues(const std::vector<std::vector<T>> &newValues) {
    values = newValues;
  }

  /// Creates a string representation of this matrix using the matrix text representation by MATLAB.
  /// For example, a 3x2 matrix would look like [2 3; 2 332; 43 3] where the semicolon (;) serves as row delimiter.
  /// \return A string representation of this matrix.
  std::string toString() override {
    std::stringstream outputStr;
    // print boolean values as text (true, false) by default, otherwise the output (0,1) cannot be distinguished from
    // the output generated by integers
    outputStr << std::boolalpha;
    // return the scalar value as string
    if (isScalar()) {
      outputStr << getScalarValue();
      return outputStr.str();
    }
    // use MATLAB's matrix style for string representation, e.g., for a 3x3 matrix: [2 2 33; 3 1 1; 3 11 9]
    const std::string elementDelimiter = " ";
    const std::string rowDelimiter = "; ";
    outputStr << "[";
    for (int i = 0; i < values.size(); ++i) {
      for (int j = 0; j < values[i].size(); ++j) {
        outputStr << values[i][j];
        if (j!=values[i].size() - 1) outputStr << elementDelimiter;
      }
      if (i!=values.size() - 1) outputStr << rowDelimiter;
    }
    outputStr << "]";
    return outputStr.str();
  }

  /// Takes a value and compares all elements of the matrix with that value. Returns True if all of the elements match
  /// the given value, otherwise returns False.
  /// \param valueToBeComparedWith The value that all elements of this matrix are compared with.
  /// \return True if all values equal the given value (valueToBeComparedWith), otherwise False.
  bool allValuesEqual(T valueToBeComparedWith) {
    for (int i = 0; i < values.size(); ++i) {
      for (int j = 0; j < values[i].size(); ++j) {
        if (values[i][j]!=valueToBeComparedWith) return false;
      }
    }
    return true;
  }

  /// Transposes a matrix, i.e., moves every element a_ij to a_ji.
  /// \param inPlace If True, then modifies the current matrix and returns the modified matrix. In case that
  ///        inPlace is False, keeps the current matrix untouched and returns instead a transposed copy.
  /// \return The transposed matrix that is either the transposed current matrix (inPlace=True) or a transposed copy of
  ///         the current matrix (inPlace=False).
  Matrix<T> *transpose(bool inPlace) {
    Matrix<T> *matrixToTranspose = inPlace ? this : new Matrix<T>(*this);
    std::vector<std::vector<T>> transposedVec(matrixToTranspose->values[0].size(), std::vector<T>());
    for (int i = 0; i < matrixToTranspose->values.size(); ++i) {
      for (int j = 0; j < matrixToTranspose->values[i].size(); ++j) {
        transposedVec[j].push_back(matrixToTranspose->values[i][j]);
      }
    }
    matrixToTranspose->getDimensions().update(matrixToTranspose->values[0].size(), matrixToTranspose->values.size());
    matrixToTranspose->values = transposedVec;
    return matrixToTranspose;
  }

  /// Rotates a matrix by a given rotationFactor.
  /// \param rotationFactor Determines the number of rotations and its direction. If rotationFactor is positive (>0),
  ///        then the elements are rotated to the right, otherwise (rotationFactor<0) the elements are rotated to the
  ///        left.
  /// \param inPlace If True, then modifies the current matrix and returns the modified matrix. In case that
  ///        inPlace is False, keeps the current matrix untouched and returns instead a transposed copy.
  /// \return The rotated matrix that is either the rotated current matrix (inPlace=True) or a rotated copy of
  ///         the current matrix (inPlace=False).
  Matrix<T> *rotate(int rotationFactor, bool inPlace) override {
    Matrix<T> *matrixToRotate = inPlace ? this : new Matrix<T>(*this);
    if (matrixToRotate->getDimensions().equals(1, -1)) {  // a row vector
      auto &vec = matrixToRotate->values[0];
      std::rotate(vec.begin(), computeRotationTarget(vec.begin(), vec.size(), rotationFactor), vec.end());
    } else if (matrixToRotate->getDimensions().equals(-1, 1)) {  // a column vector
      // Transpose the vector, rotate it, transpose it again. This is needed because std::rotate requires all elements
      // in a single vector. As our matrix is represented using row vectors, we need to transform column->row vector.
      matrixToRotate->transpose(true);
      auto &vec = matrixToRotate->values[0];
      std::rotate(vec.begin(), computeRotationTarget(vec.begin(), vec.size(), rotationFactor), vec.end());
      matrixToRotate->transpose(true);
    } else {
      throw std::invalid_argument("Rotation only supported for 1-dimensional vectors.");
    }
    return matrixToRotate;
  }

  /// Overwrites the current matrix values by the given scalarValue such that the matrix finally has the dimension
  /// specified by targetDimension and consists of same-value elements.
  /// For example, given targetDimension=(2,4) and scalarValue=7 would result in the matrix [7 7 7 7; 7 7 7 7].
  /// \param targetDimension The dimension this matrix should be expanded to.
  /// \param scalarValue The scalar value this matrix should be filled with.
  void expandAndFillMatrix(Dimension &targetDimension, T scalarValue) {
    values = std::vector<std::vector<T>>(targetDimension.numRows,
                                         std::vector<T>(targetDimension.numColumns, scalarValue));
    getDimensions().update(targetDimension.numRows, targetDimension.numColumns);
  }

  /// Clones the current matrix.
  /// \return A clone of the current matrix.
  Matrix<T> *clone() {
    // call the Matrix's copy constructor
    return new Matrix<T>(*this);
  }

  /// A helper method that is defined in AbstractMatrix and overridden here. It casts the first parameter
  /// (AbstractMatrix) to Matrix<T> where T is the template type (e.g., int, float). This is needed to call the suitable
  /// applyBinaryOperatorComponentwise method that handles the binary operator application.
  /// \param rhsOperand The operand on the right hand-side. The current matrix is the left hand-side operand.
  /// \param op The operator to be applied on the two given matrices.
  /// \return The AbstractMatrix resulting from applying the operator op on the two matrices.
  AbstractMatrix *applyBinaryOperator(AbstractMatrix *rhsOperand, Operator *op) override {
    return applyBinaryOperatorComponentwise(dynamic_cast<Matrix<T> *>(rhsOperand), op);
  }

  /// Takes two Matrix<T> objects (where T is of same type!) and applies the operator op componentwise to them.
  /// \param rhsOperand The operand on the right hand-side. The current matrix is the left hand-side operand.
  /// \param op THe operator to be aplied on the two given matrices.
  /// \return AbstractMatrix resulting from applying the operator op on the two matrices.
  AbstractMatrix *applyBinaryOperatorComponentwise(Matrix<T> *rhsOperand, Operator *op);

  /// Applies the unary operator specified by the given Operator os to every element of the matrix.
  /// \param os The operator to be applied to every element of the matrix.
  /// \return A new matrix where the Operator as was applied to every element.
  AbstractMatrix *applyUnaryOperatorComponentwise(Operator *os) override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_MATRIX_H_
