#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_MATRIX_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_MATRIX_H_

#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "AbstractMatrix.h"
#include "Dimension.h"
#include "Operator.h"
#include "LiteralFloat.h"

using json = nlohmann::json;

/// A template-based Matrix class that stores elements of type T.
/// \tparam T The type of the matrix elements.
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
  /// A matrix defined by row vectors.
  std::vector<std::vector<T>> values;

  /// The dimension of the matrix associated to this instance.
  Dimension dim;

  /// Creates a new empty matrix.
  Matrix() : values(std::vector<std::vector<T>>()), dim(Dimension(0, 0)) {}

  /// Creates a new matrix with the elements provided in inputMatrix.
  /// \param inputMatrix The elements of the matrix to create.
  Matrix(std::vector<std::vector<T>> inputMatrix)  /* NOLINT intentionally not explicit */;

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

  /// Returns the only element of the (scalar) matrix if this matrix consists of a single element only, otherwise throws
  /// an exception to notify the user that this is not a scalar.
  /// \throws std::logic_error if this matrix has more than one value.
  /// \return The single element of the matrix.
  T getScalarValue() const {
    if (isScalar()) return values[0][0];
    throw std::logic_error("getScalarValue() not allowed on non-scalars!");
  }

  /// Verifies that this matrix access is valid, i.e., matrix indices are within valid bounds.
  /// \param rowIndex The row index of the element to be accessed.
  /// \param columnIndex The column index of the element to be accessed.
  void boundCheckMatrixAccess(int rowIndex, int columnIndex) {
    if (!dim.isValidAccess(rowIndex, columnIndex)) {
      std::stringstream ss;
      ss << "Invalid matrix indices: Cannot access " << Dimension(rowIndex, columnIndex) << " ";
      ss << "because matrix has dimensions " << getDimensions() << ".";
      throw std::invalid_argument(ss.str());
    }
  }

  void checkBoundsAndResizeMatrix(int rowIdx, int colIdx) {
    if (rowIdx < 0 || colIdx < 0) {
      throw std::runtime_error("Matrix access with index < 0 is invalid! "
                               "Given indices (row, col): " + std::to_string(rowIdx) + ", " + std::to_string(colIdx));
    }

    // Resize the outer vector by adding new vectors that have the same size as the already existing ones.
    // Note: The new size must be rowIdx+1 because a vector is 0-indexed.
    if (rowIdx + 1 > values.size()) {
      if (values.empty()) {  // <==> values.size() == 0
        values.resize(rowIdx + 1);
      } else {
        values.resize(rowIdx + 1, std::vector<T>(values.at(0).size()));
      }
    }
    // Resize the inner vectors -- all of them, to ensure that all rows have the same #elements.
    if (colIdx + 1 > values.at(rowIdx).size()) { for (auto &rw : values) rw.resize(colIdx + 1); }
  }

  /// Returns a reference to the element at index specified by the given (row, column) pair.
  /// Note: Returning std::vector<T>::reference is required here. Credits to Mike Seymour from stackoverflow.com
  /// (https://stackoverflow.com/a/25770060/3017719) for pointing this out.
  /// \param row The row number of the element to be returned a reference to.
  /// \param column The column number of the element to be returned a reference to.
  /// \return A reference to the element at position (row, column).
  typename std::vector<T>::reference operator()(int row, int column) {
    boundCheckMatrixAccess(row, column);
    return values[row][column];
  }

  /// Overwrites the matrix's values by the new values given.
  /// \param newValues The values to be used to overwrite the matrix's existing values.
  void setValues(const std::vector<std::vector<T>> &newValues) {
    values = newValues;
    int elementsPerRow = values.empty() ? 0 : values.at(0).size();
    for (auto const &rowVector : values) {
      if (rowVector.size()!=elementsPerRow) {
        throw std::invalid_argument("Vector rows must all have the same number of elements!");
      }
    }
    dim.update(newValues.size(), elementsPerRow);
  }

  /// Takes a value and compares all elements of the matrix with that value. Returns True if all of the elements match
  /// the given value, otherwise returns False.
  /// \param valueToBeComparedWith The value that all elements of this matrix are compared with.
  /// \return True if all values equal the given value (valueToBeComparedWith), otherwise False.
  bool allValuesEqual(T valueToBeComparedWith) {
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values[i].size(); ++j) {
        if (values[i][j]!=valueToBeComparedWith) return false;
      }
    }
    return true;
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

  /// Takes two Matrix<T> objects (where T is of same type!) and applies the operator op componentwise to them.
  /// \param rhsOperand The operand on the right hand-side. The current matrix is the left hand-side operand.
  /// \param op THe operator to be aplied on the two given matrices.
  /// \return AbstractMatrix resulting from applying the operator op on the two matrices.
  AbstractMatrix *applyBinaryOperatorComponentwise(Matrix<T> *rhsOperand, Operator *op);

  AbstractMatrix *applyBinaryOperator(AbstractMatrix *rhsOperand, Operator *op) override {
    return applyBinaryOperatorComponentwise(dynamic_cast<Matrix<T> *>(rhsOperand), op);
  }

  [[nodiscard]] bool isScalar() const override {
    // requirements that must all be fulfilled such that a value is considered as a scalar:
    // - it is a single element, i.e., it has dimension (1,1)
    // - it is a literal (e.g., float, int, bool) or a std::string
    // - it is NOT a pointer
    //throw std::runtime_error("TODO: IMPLEMENT WITH TEMPLATE SPECIALIZATION");
    return dim.equals(1, 1) && (std::is_literal_type_v<T> || std::is_same_v<T, std::string>) && !std::is_pointer_v<T>;
  }

  [[nodiscard]] bool isEmpty() const override {
    return dim.equals(0, 0);
  }

  AbstractMatrix *applyUnaryOperatorComponentwise(Operator *os) override {
    throw std::runtime_error("applyUnaryOpComponentwise is unimplemented for type T: " + std::string(typeid(T).name()));
  }

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
    for (size_t i = 0; i < values.size(); ++i) {
      for (size_t j = 0; j < values[i].size(); ++j) {
        addElementToStringStream(values[i][j], outputStr);
        if (j!=values[i].size() - 1) outputStr << elementDelimiter;
      }
      if (i!=values.size() - 1) outputStr << rowDelimiter;
    }
    outputStr << "]";
    return outputStr.str();
  }

  Matrix<T> *transpose(bool inPlace) override {
    if (getDimensions().equals(0, 0)) return inPlace ? this : new Matrix<T>();
    Matrix<T> *matrixToTranspose = inPlace ? this : new Matrix<T>(*this);
    std::vector<std::vector<T>> transposedVec(matrixToTranspose->values[0].size(), std::vector<T>());
    for (size_t i = 0; i < matrixToTranspose->values.size(); ++i) {
      for (size_t j = 0; j < matrixToTranspose->values[i].size(); ++j) {
        transposedVec[j].push_back(matrixToTranspose->values[i][j]);
      }
    }
    matrixToTranspose->getDimensions().update(matrixToTranspose->values[0].size(), matrixToTranspose->values.size());
    matrixToTranspose->values = transposedVec;
    return matrixToTranspose;
  }

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

  std::vector<T> getNthRowVector(int rowIdx) {
    if (rowIdx < 0 || rowIdx >= getDimensions().numRows) {
      throw std::invalid_argument("getNthRowVector failed: Invalid row index given!");
    }
    return values.at(rowIdx);
  }

  std::vector<T> getNthColumnVector(int colIdx) {
    if (colIdx < 0 || colIdx >= getDimensions().numColumns) {
      throw std::invalid_argument("getNthColumnVector failed: Invalid column index given!");
    }
    std::vector<T> result;
    for (size_t i = 0; i < values.size(); ++i) result.push_back(values.at(i).at(colIdx));
    return result;
  }

  void addElementToStringStream(T elem, std::stringstream &s) {
    s << elem;
  }

  bool containsAbstractExprs() override {
    return false;
  }

  [[nodiscard]] std::string getNodeType() const override {
    return std::string("Matrix<" + std::string(typeid(T).name()) + ">");
  }

  void accept(Visitor &v) override {
    v.visit(*this);
  }

  int getMaxNumberChildren() override {
    return -1;
  }

  bool supportsCircuitMode() override {
    return true;
  }

  bool operator==(const Matrix &rhs) const;

  bool operator!=(const Matrix &rhs) const {
    return !(rhs==*this);
  }

  [[nodiscard]] json toJson() const override {
    throw std::runtime_error("toJson is unimplemented for type T: " + std::string(typeid(T).name()));
  }

  AbstractExpr *getElementAt(int row, int column) override {
    throw std::logic_error("getElementAt failed: Value in matrix is of unknown type. "
                           "Cannot determine associated AbstractLiteral subtype.");
  }

  void setElementAt(int row, int column, AbstractExpr *element) override {
    throw std::runtime_error("setElementAt is unimplemented for type T: " + std::string(typeid(T).name()));
  }

  void appendVectorAt(int idx, AbstractMatrix *mx) override {
    // determine if given matrix mx is a row or column vector
    bool isRowVector = mx->getDimensions().equals(1, -1);
    bool isColumnVector = mx->getDimensions().equals(-1, 1);

    if (!isRowVector && !isColumnVector) {
      throw std::runtime_error("Cannot executed appendVectorAt as given matrix is neither a row vector (single row) nor"
                               " a column vector (single column). Aborting!");
    }

    // If mx has dimension (1,1) then mx is a single value, i.e., we can treat it is a row vector to save the costs
    // involved in matrix transposition.
    if (isRowVector && isColumnVector) isColumnVector = false;

    // by transposing the matrix we can apply the same logic to a column vector, afterwards we transpose the
    // resulting matrix (this) back to its original form
    if (isColumnVector) {
      mx->transpose(true);
      transpose(true);
    }

    // if this matrix already contains rows, we require that the matrix to be appended has the same number of row
    // elements (= number of columns)
    if (!values.empty() && values.at(0).size()!=mx->getDimensions().getNthDimensionSize(1)) {
      std::stringstream ss;
      ss << "Dimension mismatch! Cannot execute appendVectorAt(" << idx << ") because ";
      ss << "given matrix has dimensions " << getDimensions() << " ";
      ss << "but vector to be appended has dimensions " << mx->getDimensions() << ".";
      throw std::runtime_error(ss.str());
    }

    // cast the matrix to enable accessing methods that are not exposed to AbstractMatrix (e.g., getNthRowVector)
    auto castedMx = dynamic_cast<Matrix<T> *>(mx);
    if (castedMx==nullptr) {
      throw std::runtime_error("Cast AbstractMatrix to Matrix<T> failed! Cannot execute appendVectorAt.");
    }

    // add the new row
    if (values.size()==idx) {
      // append by pushing new row at the end
      values.push_back(castedMx->getNthRowVector(0));
    } else if (values.size() < idx) {
      // resize by adding vectors of the same size as the existing ones
      if (!values.empty()) {
        values.resize(idx, std::vector<T>(values.at(0).size()));
      } else {
        values.resize(idx);
      }
      // add row by pushing it at the end
      values.push_back(castedMx->getNthRowVector(0));
    } else {
      // overwrite existing row
      values.at(idx) = castedMx->getNthRowVector(0);
    }
    // update the dimensions of this matrix
    getDimensions().update(values.size(), values.empty() ? 0 : values.at(0).size());

    // transpose this matrix back as we transposed the given matrix mx previously to avoid reimplmenting the append
    // logic for column vectors
    if (isColumnVector) transpose(true);
  }

  void replaceChild(AbstractNode *originalChild, AbstractNode *newChildToBeAdded) override {
    // values T are primitives (e.g., int, float, bool, strings) and
    AbstractNode::replaceChild(originalChild, newChildToBeAdded);
  }

  Dimension &getDimensions() override {
    return dim;
  }

  Matrix<T> *clone(bool keepOriginalUniqueNodeId) override {
    // it's sufficient to call the copy constructor that creates a copy of all primitives (int, float, etc.)
    return new Matrix<T>(*this);
  }

};

/// Compute new matrix by applying binary function f component-wise on both matrices.
/// For example, for matrices A,B with each dim (M, N), the resulting matrix would look like:
///     / f(a_00, b_00)  f(a_01, b_01)  ...  f(a_0N, b_0N) \
///     | f(a_10, b_10)      ...        ...      ...       |
///     |       ...          ...        ...      ...       |
///     \ f(a_M0, b_M0)      ...        ...  f(a_MN, b_MN) /
/// \tparam T The type of the matrix elements.
/// \param A The matrix whose elements are used as the left hand-side operand.
/// \param B The matrix whose elements are used as the right hand-side operand.
/// \param f The binary function f : (T,T) -> T to be applied componentwise on the elements of A, B.
/// \return A new Matrix<T> where each component is computed by taking elements of A and B of the same position
///         and applying the binary function f on them, i.e., f(a_{ij}, b_{i,j}).
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
  std::vector<std::vector<T>> result;
  for (size_t i = 0; i < A->values.size(); ++i) {
    result.push_back(std::vector<T>());
    for (size_t j = 0; j < A->values[0].size(); ++j) {
      result[i].push_back(f((*A)(i, j), (*B)(i, j)));
    }
  }
  return new Matrix<T>(result);
}

/// Applies a unary function f: T -> T on each element of a matrix. Compute new matrix by applying unary function f on
/// each element of matrixA, e.g., for (M, N)-matrix matrixA:
///     /  f(a_00)    f(a_01)    ...    f(a_0N)  \
///     |  f(a_10)      ...      ...      ...    |
///     |   ...         ...      ...      ...    |
///     \  f(a_M0)      ...      ...    f(a_MN)  /
/// \tparam T The type of the matrix elements.
/// \param matrixA The matrix on whose elements f should be applied to.
/// \param f The function f: T -> T to be applied on each element of matrixA.
/// \return A new matrix resulting from applying f on each element.
template<typename T>
static Matrix<T> *applyOnEachElement(Matrix<T> *matrixA, std::function<T(T)> f) {
  std::vector<std::vector<T>> result;
  for (size_t i = 0; i < matrixA->values.size(); ++i) {
    result.push_back(std::vector<T>());
    for (size_t j = 0; j < matrixA->values[0].size(); ++j) {
      result[i].push_back(f((*matrixA)(i, j)));
    }
  }
  return new Matrix<T>(result);
}

/// For matrices A with dim (M,N) and B with dim (N,P) the following algorithm computes C with dim (M,P) where
///   c_ij = sum_{k=1}^{n} a_ik * b_kj.
/// This is basically the summation over multiplying each element in the row-vector from A with each element in
/// the column-vector from B. See, for example, Wikipedia for more details and images:
///   https://en.wikipedia.org/wiki/Matrix_multiplication.
/// Note that the matrix multiplication is not commutative, i.e., AxB != BxA.
/// \tparam T The type of the Matrix elements.
/// \param matrixA The left hand-side operand of the matrix multiplication.
/// \param matrixB The right hand-side operand of the matrix multiplication.
/// \return A matrix of same type with the values as defined by the matrix multiplication.
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
  std::vector<std::vector<T>> result(
      matrixA->getDimensions().numRows, std::vector<T>(matrixB->getDimensions().numColumns));

  // matrixA has dim (M,N), matrixB has dim (N,P)
  int dimM = matrixA->getDimensions().numRows;
  int dimN = matrixA->getDimensions().numColumns;
  int dimP = matrixB->getDimensions().numColumns;
  T sum = 0;
  // Very inefficient: O(n^3)! Replace it, for example, by Strassen's multiplication algorithm.
  for (int k = 0; k < dimN; ++k) {
    for (int i = 0; i < dimM; ++i) {
      for (int j = 0; j < dimP; ++j) {
        result[i][j] += (*matrixA)(i, k)*(*matrixB)(k, j);
      }
    }
  }
  return new Matrix<T>(result);
}

// Because of a bug in the Microsoft Visual C++ Compiler (MSVC),
// code for template specialisations is not properly emitted if the functions are defined inside the class body
// and the specialized version isn't ODR-used in the same translation unit
// Therefore, these following functions are all defined outside the class:

template<class T>
Matrix<T>::Matrix(std::vector<std::vector<T>> inputMatrix)  /* NOLINT intentionally not explicit */
    : values(std::move(inputMatrix)), dim(values.size(), values.empty() ? 0 : values.at(0).size()) {
  int elementsPerRow = values.empty() ? 0 : values.at(0).size();
  for (auto const &rowVector : values) {
    if (rowVector.size()!=elementsPerRow) {
      throw std::invalid_argument("Vector rows must all have the same number of elements!");
    }
  }
}

template<class T>
AbstractMatrix *Matrix<T>::applyBinaryOperatorComponentwise(Matrix<T> *rhsOperand, Operator *op) {
  throw std::runtime_error(
      "applyOperatorComponentwise is unimplemented for type T: " + std::string(typeid(T).name()));
}
template<class T>
bool Matrix<T>::operator==(const Matrix &rhs) const {
  return values==rhs.values && dim==rhs.dim;
}

// declarations of specific specialisations

template<>
bool Matrix<AbstractExpr *>::operator==(const Matrix &rhs) const;

template<>
AbstractMatrix *Matrix<int>::applyBinaryOperatorComponentwise(Matrix<int> *rhsOperand, Operator *os);

template<>
AbstractMatrix *Matrix<float>::applyBinaryOperatorComponentwise(Matrix<float> *rhsOperand, Operator *os);

template<>
AbstractMatrix *Matrix<bool>::applyBinaryOperatorComponentwise(Matrix<bool> *rhsOperand, Operator *os);

template<>
AbstractMatrix *Matrix<std::string>::applyBinaryOperatorComponentwise(Matrix<std::string> *rhsOperand, Operator *os);

template<>
AbstractMatrix *Matrix<int>::applyUnaryOperatorComponentwise(Operator *os);

template<>
AbstractMatrix *Matrix<float>::applyUnaryOperatorComponentwise(Operator *os);

template<>
AbstractMatrix *Matrix<bool>::applyUnaryOperatorComponentwise(Operator *os);

template<>
AbstractExpr *Matrix<int>::getElementAt(int row, int column);

template<>
AbstractExpr *Matrix<float>::getElementAt(int row, int column);

template<>
AbstractExpr *Matrix<bool>::getElementAt(int row, int column);

template<>
AbstractExpr *Matrix<std::string>::getElementAt(int row, int column);

template<>
AbstractExpr *Matrix<AbstractExpr *>::getElementAt(int row, int column);

template<>
Matrix<AbstractExpr *>::Matrix(std::vector<std::vector<AbstractExpr *>> inputMatrix);

template<>
[[nodiscard]] json Matrix<AbstractExpr *>::toJson() const;

template<>
json Matrix<bool>::toJson() const;

template<>
json Matrix<int>::toJson() const;

template<>
json Matrix<float>::toJson() const;

template<>
json Matrix<double>::toJson() const;

template<>
json Matrix<std::string>::toJson() const;

template<>
Matrix<AbstractExpr *> *Matrix<AbstractExpr *>::clone(bool keepOriginalUniqueNodeId);

template<>
void Matrix<AbstractExpr *>::addElementToStringStream(AbstractExpr *elem, std::stringstream &s);

template<>
void Matrix<AbstractExpr *>::setElementAt(int row, int column, AbstractExpr *element);

template<>
void Matrix<int>::setElementAt(int row, int column, AbstractExpr *element);

template<>
void Matrix<float>::setElementAt(int row, int column, AbstractExpr *element);

template<>
void Matrix<bool>::setElementAt(int row, int column, AbstractExpr *element);

template<>
void Matrix<std::string>::setElementAt(int row, int column, AbstractExpr *element);

template<>
void Matrix<AbstractExpr *>::replaceChild(AbstractNode *originalChild, AbstractNode *newChildToBeAdded);

template<>
bool Matrix<AbstractExpr *>::containsAbstractExprs();

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_MATRIX_H_