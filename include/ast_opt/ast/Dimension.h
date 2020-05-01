#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_DIMENSION_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_DIMENSION_H_

#include <iostream>

class Dimension {
 public:
  int numRows, numColumns;

  Dimension(int numberOfRows, int numberOfColumns);

  bool operator==(const Dimension &rhs) const;

  bool operator!=(const Dimension &rhs) const;

  friend std::ostream &operator<<(std::ostream &os, const Dimension &dimension);

  bool isValidAccess(int row, int column);

  [[nodiscard]] bool equals(int rows, int columns) const;

  void update(int numberOfRows, int numberOfColumns);

  int getNthDimensionSize(int n);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_DIMENSION_H_
