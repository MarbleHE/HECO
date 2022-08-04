#include <vector>
#include <stdexcept>

/// Original, plain C++ program for a dot product between two vectors
///
/// \param x vector of size n
/// \param y vector of size n
/// \return <x,y> dot product of x and y
int dotProduct(const std::vector<int> &x, const std::vector<int> &y) {

  if (x.size()!=y.size()) throw std::runtime_error("Vectors in dot product must have the same length.");
  int sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sum += x[i]*y[i];
  }
  return sum;
}