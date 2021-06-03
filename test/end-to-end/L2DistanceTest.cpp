#include <vector>
#include <stdexcept>

/// Original, plain C++ program for the (squared) L2 distance between two vectors
///
/// Since sqrt() is not supported in BFV, we omit it and compute the squared distance
/// \param x vector of size n
/// \param y vector of size n
/// \return L2 distance between the two vectors
int squaredL2Distance(const std::vector<int> &x, const std::vector<int> &y) {

  if (x.size()!=y.size()) throw std::runtime_error("Vectors in L2 distance must have the same length.");
  int sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sum += (x[i] - y[i])*(x[i] - y[i]);
  }
  return sum;
}
