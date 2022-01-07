#include <vector>
#include <stdexcept>
#include "ast_opt/ast_utilities/Scope.h"
#include "ast_opt/ast_parser/Parser.h"
#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV
#include "bench/HammingDistance.h"
#endif

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

#ifdef HAVE_SEAL_BFV
TEST(HammingDistanceTest, NaiveClear_Porcupine_Equivalence) { /* NOLINT */
  // Create two vectors of bits (booleans),
  // TODO: Create test values from fixed random seed
  std::vector<bool> a(4, 0);
  std::vector<bool> b(4, 1);

  MultiTimer dummy = MultiTimer();
  auto result = encryptedHammingDistancePorcupine(dummy, a, b, 2 << 13);

  // Compare to reference cleartext implementation
  EXPECT_EQ(hammingDistance(a, b), result);
}

TEST(HammingDistanceTest, NaiveClear_Batched_Equivalence) { /* NOLINT */
  // Create two vectors of bits (booleans),
  // TODO: Create test values from fixed random seed
  std::vector<bool> a(4, 0);
  std::vector<bool> b(4, 1);

  MultiTimer dummy = MultiTimer();
  auto result = encryptedBatchedHammingDistance(dummy, a, b, 2 << 13);

  // Compare to reference cleartext implementation
  EXPECT_EQ(hammingDistance(a, b), result);
}

#endif //HAVE_SEAL_BFV

//Note: Hamming distance over binary vectors can be computed efficiently in Z_p by using NEQ = XOR = (a-b)^2