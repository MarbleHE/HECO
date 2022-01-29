#include <vector>
#include <stdexcept>
#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV
#include "L2Distance.h"
#endif

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

#ifdef HAVE_SEAL_BFV
TEST(SquaredL2DistanceTest, Clear_EncryptedPorcupine_Equivalence) { /* NOLINT */
  std::vector<int> a(4, 50); // TODO: Create from fixed random seed
  std::vector<int> b(4, 40); // TODO: Create from fixed random seed

  MultiTimer dummy = MultiTimer();
  auto result = encryptedL2DistanceSquared_Porcupine(dummy, a, b, 2 << 13);

  // Compare to reference cleartext implementation
  EXPECT_EQ(squaredL2Distance(a, b), result);
}

#endif //HAVE_SEAL_BFV