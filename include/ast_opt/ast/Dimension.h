#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_DIMENSION_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_DIMENSION_H_

#include <iostream>

class Dimension {
 public:
  int numRows, numColumns;

  Dimension() = default;

  Dimension(int numberOfRows, int numberOfColumns);

  Dimension(const Dimension&) = default;

  Dimension(Dimension&&) = default;

  Dimension& operator=(const Dimension&) = default;

  Dimension& operator=(Dimension&&) = default;

  bool operator==(const Dimension &rhs) const;

  bool operator!=(const Dimension &rhs) const;

  friend std::ostream &operator<<(std::ostream &os, const Dimension &dimension);

  bool isValidAccess(int row, int column);

  [[nodiscard]] bool equals(int rows, int columns) const;

  void update(int numberOfRows, int numberOfColumns);

  int getNthDimensionSize(int n) const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_DIMENSION_H_
