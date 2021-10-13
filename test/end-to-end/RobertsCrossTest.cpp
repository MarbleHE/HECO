#include <cmath>
#include <random>
#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV
#include "bench/RobertsCross.h"
#endif

/// Original, plain C++ program for a naive RobertsCross Kernel
/// This uses two 2x2 Kernels, which we pad to 3x3 kernels for ease of implementation
///         |  1   0  0 |
///   w1 =  |  0  -1  0 |
///         |  0   0  0 |
///
///         |  0  +1  0 |
///   w2 =  | -1   0  0 |
///         |  0   0  0 |
/// This uses wrap-around padding
/// and computes sqt((w1 * I)^2 + (w2 * I)^2) where * stands for convolution
///
/// To avoid square roots (which are unsupported in BFV) we return the squared result
/// The client can easily apply sqrt() to the result after decryption
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> naiveRobertsCrossKernel(const std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));

  // Compute first Kernel

  // Encoded same as images
  std::vector<std::vector<int>> weightMatrix1 = {{0, 0, 1}, {0, -1, 0}, {0, 0, 0}};
  std::vector<int> img2(img.begin(), img.end());
  for (int x = 0; x < imgSize; ++x) {
    for (int y = 0; y < imgSize; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
          value = value + weightMatrix1.at(i + 1).at(j + 1)
              *img.at(((x + i)*imgSize + (y + j))%img.size());
        }
      }
      img2[imgSize*x + y] = value;
    }
  }

  // Compute Second Kernel

  // Encoded same as images
  std::vector<std::vector<int>> weightMatrix2 = {{0, -1, 0}, {0, 0, 1}, {0, 0, 0}};
  std::vector<int> img3(img.begin(), img.end());
  for (int x = 0; x < imgSize; ++x) {
    for (int y = 0; y < imgSize; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
          value = value + weightMatrix2.at(i + 1).at(j + 1)
              *img.at(((x + i)*imgSize + (y + j))%img.size());
        }
      }
      img3[imgSize*x + y] = value;
    }
  }

  std::vector<int> img4(img.begin(), img.end());
  // Compute x^2 + y^2, square root omitted
  for (int x = 0; x < imgSize; ++x) {
    for (int y = 0; y < imgSize; ++y) {
      img4[x*imgSize + y] =
          img2.at(x*imgSize + y)*img2.at(x*imgSize + y) + img3.at(x*imgSize + y)*img3.at(x*imgSize + y);
    }
  }
  return img4;
}

/// Original, plain C++ program for a faster RobertsCross Kernel
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> fastRobertsCrossKernel(const std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));

  //TODO: Is it possible (and useful) to apply Kernel separation to Roberts Cross?
  return img;
}


// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838
// use a 4x4 matrix for the tests
#define MATRIX_SIZE 4

class RobertsCrossKernelTest : public ::testing::Test {  /* NOLINT (predictable sequence expected) */
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

  void getInputMatrix(size_t size, std::vector<std::vector<int>> &destination) {
    // reset the RNG to make sure that every call to this method results in the same numbers
    resetRandomEngine();
    // make sure we clear desination vector before, otherwise resize could end up appending elements
    destination.clear();
    destination.resize(size, std::vector<int>(size));
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        destination[i][j] = myUnifIntDist(randomEngine);
      }
    }
  }

  void getInputMatrix(size_t size, std::vector<int> &destination) {
    // make sure we clear desination vector before, otherwise resize could end up appending elements
    destination.clear();
    std::vector<std::vector<int>> data;
    getInputMatrix(size, data);
    std::size_t total_size = 0;
    for (const auto &sub : data) total_size += sub.size();
    destination.reserve(total_size);
    for (const auto &sub : data) destination.insert(destination.end(), sub.begin(), sub.end());
  }

  void printMatrix(size_t size, std::vector<int> &matrix) {
    for (int64_t row = (int64_t) size - 1; row >= 0; --row) {
      std::cout << std::setw(8) << matrix.at(0*size + row);
      for (size_t col = 1; col < size; ++col) {
        std::cout << std::setw(8) << matrix.at(col*size + row);
      }
      std::cout << std::endl;
    }
  }
};

#ifdef HAVE_SEAL_BFV
TEST_F(RobertsCrossKernelTest, Clear_EncryptedPorcupine_Equivalence) { /* NOLINT */
  size_t poly_modulus_degree = 2 << 12;
  size_t img_size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  getInputMatrix(img_size, img);

  MultiTimer dummy = MultiTimer();
  auto result = encryptedRobertsCrossPorcupine(dummy, img, poly_modulus_degree);
  result.resize(img.size());
  std::vector<int> enc(begin(result), end(result));

  // Compare to reference cleartext implementation
  auto ref = naiveRobertsCrossKernel(img);
  EXPECT_EQ(enc, ref);
}

TEST_F(RobertsCrossKernelTest, Clear_EncryptedBatched_Equivalence) { /* NOLINT */
  size_t poly_modulus_degree = 2 << 12;
  size_t img_size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  getInputMatrix(img_size, img);
  std::cout << "img:" << std::endl;
  printMatrix(img_size, img);

  MultiTimer dummy = MultiTimer();
  auto result = encryptedBatchedRobertsCross(dummy, img, poly_modulus_degree);
  result.resize(img.size());
  std::vector<int> enc(begin(result), end(result));

  // Compare to reference cleartext implementation
  auto ref = naiveRobertsCrossKernel(img);
  std::cout << "naive:" << std::endl;
  printMatrix(img_size, ref);
  EXPECT_EQ(enc, ref);
}

TEST_F(RobertsCrossKernelTest, Clear_EncryptedNaive_Equivalence) { /* NOLINT */
  size_t poly_modulus_degree = 2 << 12;
  size_t img_size = 8;
  std::vector<int> img;
  getInputMatrix(img_size, img);
  //std::cout << "img:" << std::endl;
  //printMatrix(img_size, img);

  MultiTimer dummy = MultiTimer();
  auto result = encryptedNaiveRobertsCross(dummy, img, poly_modulus_degree);
  //std::cout << "encrypted:" << std::endl;
  //printMatrix(img_size, result);

  // Compare to reference cleartext implementation
  auto ref = naiveRobertsCrossKernel(img);
  //std::cout << "clear:" << std::endl;
  //printMatrix(img_size, ref);

  EXPECT_EQ(result, ref);
}
#endif //HAVE_SEAL_BFV

/// Test to ensure that naiveRobertsCrossKernel and fastRobertsCrossKernel actually compute the same thing!
TEST_F(RobertsCrossKernelTest, DISABLED_NaiveRobertsCrossKernel_FastRobertsCrossKernel_Equivalence) {  /* NOLINT */

  size_t size = 4;
  std::vector<int> img;
  RobertsCrossKernelTest::getInputMatrix(size, img);
  std::cout << "img:" << std::endl;
  printMatrix(size, img);

  auto naive = naiveRobertsCrossKernel(img);
  std::cout << "naive:" << std::endl;
  printMatrix(size, naive);

  auto fast = fastRobertsCrossKernel(img);
  std::cout << "fast:" << std::endl;
  printMatrix(size, fast);

  EXPECT_EQ(fast, naive);
}