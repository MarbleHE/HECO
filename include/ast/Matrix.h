#ifndef AST_OPTIMIZER_INCLUDE_AST_MATRIX_H_
#define AST_OPTIMIZER_INCLUDE_AST_MATRIX_H_

#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Dimension {
 public:
  int numRows, numColumns;

  Dimension(int numberOfRows, int numberOfColumns) : numRows(numberOfRows), numColumns(numberOfColumns) {}

  bool operator==(const Dimension &rhs) const {
    return numRows==rhs.numRows && numColumns==rhs.numColumns;
  }

  bool operator!=(const Dimension &rhs) const {
    return !(rhs==*this);
  }

  friend std::ostream &operator<<(std::ostream &os, const Dimension &dimension) {
    return os << "(" << dimension.numRows << "," << dimension.numColumns << ")";
  }

  bool isValidAccess(int row, int column) {
    // note that indices row/column start by 0
    return row < numRows && column < numColumns;
  }

  [[nodiscard]] bool equals(int rows, int columns) const {
    if (rows==-1 && columns > 0) {
      // ignore rows (-1), compare columns only
      return numColumns==columns;
    } else if (rows > 0 && columns==-1) {
      // ignore columns (-1), compare rows only
      return numRows==rows;
    } else {
      return (*this)==Dimension(rows, columns);
    }
  }

  void update(int numberOfRows, int numberOfColumns) {
    numRows = numberOfRows;
    numColumns = numberOfColumns;
  }
};

/// A helper class that allows to define all Matrix<T> specializations using a unified interface.
class CMatrix {
 public:
//  virtual void rotate(int rotationFactor) = 0;

  virtual CMatrix *rotate(int rotationFactor, bool inPlace) = 0;

  virtual Dimension &getDimensions() = 0;

  virtual std::string toString() = 0;

  [[nodiscard]] virtual bool isScalar() const = 0;

  [[nodiscard]] virtual json toJson() const = 0;
};

template<typename T>
class Matrix : public CMatrix {
 private:
  template<typename Iterator>
  Iterator computeRotationTarget(Iterator itBegin, size_t vectorSize, int rotationFactor) {
    if (rotationFactor > 0) {
      return (itBegin + vectorSize - rotationFactor);
    } else {
      return (itBegin - rotationFactor);
    }
  }

 public:
  // a matrix of row vectors
  std::vector<std::vector<T>> values;
  Dimension dim;

  Matrix(std::vector<std::vector<T>> inputMatrix)  /* NOLINT intentionally not explicit */
      : values(std::move(inputMatrix)), dim(Dimension(values.size(), values.at(0).size())) {
    int elementsPerRow = values.at(0).size();
    for (auto const &rowVector : values) {
      if (rowVector.size()!=elementsPerRow) {
        throw std::invalid_argument("Vector rows must all have the same number of elements!");
      }
    }
  }

  Matrix(T scalarValue) : dim({1, 1}) {  /* NOLINT intentionally not explicit */
    values = {{scalarValue}};
  }

  Matrix(Matrix<T> &other) : dim(Dimension(other.getDimensions().numRows, other.getDimensions().numColumns)) {
    values = other.values;
  }

  [[nodiscard]] bool isScalar() const override {
    return dim.equals(1, 1);
  }

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

  [[nodiscard]] json toJson() const override {
    // return the scalar value if this is a (1,1) scalar matrix
    if (isScalar()) return json(getScalarValue());
    // if this is a matrix, return an array of arrays like [ [a00, b01, c02], [d10, e11, f12] ]
    json arrayOfArrays = json::array();
    for (int i = 0; i < values.size(); ++i) {
      arrayOfArrays.push_back(json(values[i]));
    }
    return arrayOfArrays;
  }

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

  T getElement(int row, int column) {
    return values[row][column];
  }

  Dimension &getDimensions() override {
    return dim;
  }

  void setValues(const std::vector<std::vector<T>> &newValues) {
    values = newValues;
  }

  std::vector<T> getValuesAsVector() {
    // returns all matrix values as single one-dimensional vector
    std::vector<T> resultVec;
    for (auto &vec : values) {
      resultVec.insert(resultVec.end(), vec.begin(), vec.end());
    }
    return resultVec;
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
    // use MATLAB' matrice style for matrices, e.g., for a 3x3 matrix: [2 2 33; 3 1 1; 3 11 9]
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

  bool allValuesEqual(T valueToBeComparedWith) {
    // this is inefficient -> requires O(n^2) runtime
    for (int i = 0; i < values.size(); ++i) {
      for (int j = 0; j < values[i].size(); ++j) {
        if (values[i][j]!=valueToBeComparedWith) return false;
      }
    }
    return true;
  }

  Matrix<T> *clone() {
    // call the Matrix's copy constructor
    return new Matrix<T>(*this);
  }
//
//  void transpose() {
//    std::vector<std::vector<T>> transposedVec(values[0].size(), std::vector<T>());
//    // this is inefficient -> requires O(n^2) runtime
//    for (int i = 0; i < values.size(); ++i) {
//      for (int j = 0; j < values[i].size(); ++j) {
//        transposedVec[j].push_back(values[i][j]);
//      }
//    }
//    dim.updateDimension(values[0].size(), values.size());
//    values = transposedVec;
//  }


  Matrix<T> *transpose(bool inPlace) {
    Matrix<T> *matrixToTranspose = inPlace ? this : new Matrix<T>(*this);
    std::vector<std::vector<T>> transposedVec(matrixToTranspose->values[0].size(), std::vector<T>());
    // this is inefficient -> requires O(n^2) runtime
    for (int i = 0; i < matrixToTranspose->values.size(); ++i) {
      for (int j = 0; j < matrixToTranspose->values[i].size(); ++j) {
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
    } else if (getDimensions().equals(-1, 1)) {  // a column vector
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
};

#endif //AST_OPTIMIZER_INCLUDE_AST_MATRIX_H_
