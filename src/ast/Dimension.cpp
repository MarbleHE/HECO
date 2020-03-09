#include "Dimension.h"

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
  // note that indices row/column start by 0
  return row < numRows && column < numColumns;
}
