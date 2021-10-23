#include <random>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "bench/MultiTimer.h"
#include "bench/LinearPolynomial.h"

/// Linear polynomial (from porcupine)
/// Calculates y - (ax^2 + bx + c)
std::vector<int> linearPolynomial(
        std::vector<int> a, std::vector<int> b, std::vector<int> x, std::vector<int> y)
{
  std::vector<int> result(a.size());
  for (int i = 0; i < a.size(); ++i) {
    // y - ax - b
    result[i] = y[i] - a[i] * x[i]- b[i];
  }

  return result;
}

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838

class LinearPolynomialTest : public ::testing::Test {  /* NOLINT (predictable sequence expected) */
 protected:
  std::default_random_engine randomEngine;
  std::uniform_int_distribution<int> myUnifIntDist;

  void SetUp() override {
    randomEngine = std::default_random_engine(RAND_SEED);  /* NOLINT (predictable sequence expected) */
    // the supported number range must be according to the FHE scheme parameters to not wrap around the modulo
    myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);
  }

 public:
  void resetRandomEngine() {
    randomEngine.seed(RAND_SEED);
  }

  void getRandomVector(std::vector<int> &destination) {
    for (int & i : destination) {
      i = myUnifIntDist(randomEngine);
    }
  }

  void printVector(std::vector<int> &vec) {
    for(int i : vec) {
      std::cout << std::setw(12) << i;
    }
    std::cout << std::endl;
  }
};

#ifdef HAVE_SEAL_BFV
TEST_F(LinearPolynomialTest, Clear_EncryptedProcupine_Equivalence) { /* NOLINT */
  size_t poly_modulus_degree = 2 << 12;
  size_t vec_size = poly_modulus_degree / 2;
  std::vector<int> x(vec_size);
  std::vector<int> y(vec_size);
  std::vector<int> a(vec_size);
  std::vector<int> b(vec_size);

  LinearPolynomialTest::getRandomVector(x);
  LinearPolynomialTest::getRandomVector(y);
  LinearPolynomialTest::getRandomVector(a);
  LinearPolynomialTest::getRandomVector(b);

  MultiTimer dummy = MultiTimer();
  auto encrypted = encryptedLinearPolynomialPorcupine(dummy, a, b, x, y, poly_modulus_degree);
  encrypted.resize(a.size());
  std::vector<int> enc(begin(encrypted), end(encrypted));

  auto clear = linearPolynomial(a, b, x, y);
  EXPECT_EQ(clear, enc);
}

TEST_F(LinearPolynomialTest, Clear_EncryptedNaive_Equivalence) { /* NOLINT */
  size_t poly_modulus_degree = 2 << 12;
  size_t vec_size = 32;
  std::vector<int> x(vec_size);
  std::vector<int> y(vec_size);
  std::vector<int> a(vec_size);
  std::vector<int> b(vec_size);

  LinearPolynomialTest::getRandomVector(x);
  LinearPolynomialTest::getRandomVector(y);
  LinearPolynomialTest::getRandomVector(a);
  LinearPolynomialTest::getRandomVector(b);

  MultiTimer dummy = MultiTimer();
  auto encrypted = encryptedLinearPolynomialNaive(dummy, a, b, x, y, poly_modulus_degree);

  auto clear = linearPolynomial(a, b, x, y);
  EXPECT_EQ(clear, encrypted);
}


#endif//HAVE_SEAL_BFV