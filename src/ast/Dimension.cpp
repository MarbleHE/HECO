#include "ast_opt/ast/Dimension.h"

Dimension::Dimension(int numberOfRows, int numberOfColumns) : numRows(numberOfRows), numColumns(numberOfColumns) {}

bool Dimension::operator==(const Dimension &rhs) const {
  return numRows==rhs.numRows && numColumns==rhs.numColumns;
}

bool Dimension::operator!=(const Dimension &rhs) const {
  return !(rhs==*this);
}

std::ostream &operator<<(std::ostream &os, const Dimension &dimension) {
  return os << "(" << dimension.numRows << "," << dimension.numColumns << ")";
}

bool Dimension::equals(int rows, int columns) const {
  if (rows==-1 && columns > 0) {
    // ignore rows (-1), compare columns only
    return numColumns==columns;
  } else if (rows > 0 && columns==-1) {
    // ignore columns (-1), compare rows only
    return numRows==rows;
  } else {
    // compare number of rows and columns using equality operator
    return (*this)==Dimension(rows, columns);
  }
}

void Dimension::update(int numberOfRows, int numberOfColumns) {
  numRows = numberOfRows;
  numColumns = numberOfColumns;
}

bool Dimension::isValidAccess(int row, int column) {
  // note that row/column indices start by 0, that is a dim (M, N) matrix has row indices 0, ..., M-1 and column indices
  // 0, ..., N-1
  return row >= 0 && row < numRows && column >= 0 && column < numColumns;
}

int Dimension::getNthDimensionSize(int n) const {
  switch (n) {
    case 0:return numRows;
    case 1:return numColumns;
    default:return 0;
  }
}
