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
 private:
  int numRows, numColumns;

 public:
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

  [[nodiscard]] bool hasDimension(int row, int column) const {
    return (*this)==Dimension(row, column);
  }
};

template<typename T>
class Matrix {
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

  [[nodiscard]] bool isScalar() const {
    return dim.hasDimension(1, 1);
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

  [[nodiscard]] json toJson() const {
    // return the scalar value if this is a (1,1) scalar matrix
    if (isScalar()) return getScalarValue();
    // if this is a matrix, return an array of arrays like [ [a00, b01, c02], [d10, e11, f12] ]
    json arrayOfArrays = json::array();
    for (int i = 0; i < values.size(); ++i) {
      arrayOfArrays.push_back(json(values[i]));
    }
    return arrayOfArrays;
  }

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

  Dimension getDimensions() {
    return dim;
  }

  void setValues(const std::vector<std::vector<int>> &newValues) {
    values = newValues;
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_MATRIX_H_
