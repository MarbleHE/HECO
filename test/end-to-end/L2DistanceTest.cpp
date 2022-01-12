#include <vector>
#include <stdexcept>
#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV
#include "bench/L2Distance.h"
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
  size_t poly_modulus_degree = 2 << 12;
  std::vector<int> a{10, 20, 10, 5}; // TODO: Create from fixed random seed
  std::vector<int> b{3, 9, 4, 15}; // TODO: Create from fixed random seed

  MultiTimer dummy = MultiTimer();
  auto result = encryptedL2DistanceSquared_Porcupine(dummy, a, b, poly_modulus_degree);

  // Compare to reference cleartext implementation
  EXPECT_EQ(squaredL2Distance(a, b), result);
}

TEST(SquaredL2DistanceTest, Clear_EncryptedNaive_Equivalence) { /* NOLINT */
  size_t poly_modulus_degree = 2 << 12;
  std::vector<int> a{10, 20, 10, 5}; // TODO: Create from fixed random seed
  std::vector<int> b{3, 9, 4, 15}; // TODO: Create from fixed random seed

  MultiTimer dummy = MultiTimer();
  auto result = encryptedL2DistanceSquared_Naive(dummy, a, b, poly_modulus_degree);

  // Compare to reference cleartext implementation
  EXPECT_EQ(squaredL2Distance(a, b), result);
}

TEST(SquaredL2DistanceTest, Clear_EncryptedBatched_Equivalence) { /* NOLINT */
  size_t poly_modulus_degree = 2 << 12;
  std::vector<int> a{10, 20, 10, 5}; // TODO: Create from fixed random seed
  std::vector<int> b{3, 9, 4, 15}; // TODO: Create from fixed random seed

  MultiTimer dummy = MultiTimer();
  auto result = encryptedBatchedSquaredL2Distance(dummy, a, b, poly_modulus_degree);

  // Compare to reference cleartext implementation
  EXPECT_EQ(squaredL2Distance(a, b), result);
}

#endif //HAVE_SEAL_BFV