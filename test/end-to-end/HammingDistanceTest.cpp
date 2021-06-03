#include <vector>
#include <stdexcept>

/// Original, plain C++ program for the hamming distance between two vectors
///
/// \param x vector of size n
/// \param y vector of size n
/// \return hamming distance between the two vectors
int hammingDistance(const std::vector<bool> &x, const std::vector<bool> &y) {

  if (x.size()!=y.size()) throw std::runtime_error("Vectors  in hamming distance must have the same length.");
  int sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sum += x[i]!=y[i];
  }
  return sum;
}

//Note: Hamming distance over binary vectors can be computed efficiently in Z_p by using NEQ = XOR = (a-b)^2