#include <vector>
#include <stdexcept>
#include "ast_opt/ast_utilities/Scope.h"
#include "ast_opt/ast_parser/Parser.h"
#include "gtest/gtest.h"

/// Original, plain C++ program for a matrix-vector product
///
/// \param matrix matrix of size m x n (row-major)
/// \param vec vector of size n
/// \return matrix * vec
std::vector<int> matrixVectorProduct(const std::vector<std::vector<int>> &matrix, const std::vector<int> &vec) {

  if (matrix.empty() || matrix[0].size()!=vec.size())
    throw std::runtime_error("Vectors  in dot product must have the same length.");

  size_t m = matrix.size();
  size_t n = vec.size();

  std::vector<int> result(matrix.size());
  for (size_t i = 0; i < m; ++i) {
    int sum = 0;
    for (size_t j = 0; j < n; ++j) {
      sum += matrix[i][j]*vec[j];
    }
    result[i] = sum;
  }
  return result;
}