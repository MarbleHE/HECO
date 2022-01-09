#include <cmath>
#include <random>
#include "abc/ast_utilities/Scope.h"
#include "abc/ast_parser/Parser.h"
#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV
#include "bench/BoxBlur.h"
#endif

/// Original, plain C++ program for a naive Box blur
/// This uses a 3x3 Kernel and applies it by sliding across the 2D image
///             | 1  1  1 |
///   w = 1/9 * | 1  1  1 |
///             | 1  1  1 |
/// This uses wrap-around padding
/// To avoid division (which is unsupported in BFV) we omit the 1/9
/// The client can easily divide the result by nine after decryption
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> naiveBoxBlur(const std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  std::vector<std::vector<int>> weightMatrix = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
  std::vector<int> img2(img.begin(), img.end());
  for (int x = 0; x < imgSize; ++x) {
    for (int y = 0; y < imgSize; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
          value += weightMatrix.at(i + 1).at(j + 1)
              *img.at(((x + i)*imgSize + (y + j))%img.size());
        }
      }
      img2[imgSize*x + y] = value;
    }
  }
  return img2;
}

/// Modified, plain C++ program for a naive Box blur
/// This uses a 2x2 Kernel and applies it by sliding across the 2D image
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> naiveBoxBlur2x2(const std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  std::vector<std::vector<int>> weightMatrix = {{1, 1}, {1, 1}};
  std::vector<int> img2(img.begin(), img.end());
  for (int x = 0; x < imgSize; ++x) {
    for (int y = 0; y < imgSize; ++y) {
      int value = 0;
      for (int j = -1; j < 1; ++j) {
        for (int i = -1; i < 1; ++i) {
          value += weightMatrix.at(i + 1).at(j + 1) * img.at(((x + i)*imgSize + (y + j)) % img.size());
        }
      }
      img2[imgSize*x + y] = value;
    }
  }
  return img2;
}

/// Original, plain C++ program for a fast Box blur
/// Instead of using a using a 2D Kernel, this uses two 1D Kernels
///         | 1 |                         | 1  1  1 |
///   1/3 * | 1 | * 1/3 [1  1  1] = 1/9 * | 1  1  1 |
///         | 1 |                         | 1  1  1 |
/// This uses wrap-around padding
///
/// This version is based http://amritamaz.net/blog/understanding-box-blur
/// The separation of the kernels allows it to do horizontal and vertical blurs separately.
/// It also uses the fact that the kernel weights are the same in each position
/// and simply adds and removes pixels from a running value computation.
///
/// To avoid division (which is unsupported in BFV) we omit the two 1/3
/// The client can easily divide the result by nine after decryption
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> fastBoxBlur(const std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  std::vector<int> img2(img.begin(), img.end());

  // Horizontal Kernel: for each row y
  for (int y = 0; y < imgSize; ++y) {
    // Get kernel for first pixel of row y, using padding
    int value = img.at((-1*imgSize + y)%img.size()) + img.at(0*imgSize + y) + img.at(1*imgSize + y);
    // Division that would usually happen here is omitted
    img2[0*imgSize + y] = value;

    // Go through the rest of row y
    for (int x = 1; x < imgSize; ++x) {
      // remove the previous pixel
      //x = middle of current kernel, x-2 = one to the left of kernel
      value -= img.at(((x - 2)*imgSize + y)%img.size());
      // add the new pixel
      //x = right pixel of previous kernel, x+1 = right pixel of new kernel
      value += img.at(((x + 1)*imgSize + y)%img.size());
      // save result
      img2[x*imgSize + y] = value;
    }

  }

  // Now apply the vertical kernel to img2

  // Create new output image
  std::vector<int> img3(img2.begin(), img2.end());

  // Vertical Kernel: for each column x
  for (int x = 0; x < imgSize; ++x) {
    // Get kernel for first pixel of column x with padding
    int value = img2.at((x*imgSize - 1)%img.size()) + img2.at(x*imgSize + 0) + img2.at(x*imgSize + 1);
    // Division that would usually happen here is omitted
    img3[x*imgSize + 0] = value;

    // Go through the rest of column x
    for (int y = 1; y < imgSize; ++y) {
      // remove the previous pixel
      //y = middle of current kernel, y-2 = one to the left of kernel
      value -= img2.at((x*imgSize + y - 2)%img.size());
      // add the new pixel
      //y = right pixel of previous kernel, y+1 = right pixel of new kernel
      value += img2.at((x*imgSize + y + 1)%img.size());
      // save result
      img3[x*imgSize + y] = value;
    }

  }
  return img3;
}

std::vector<int> fastBoxBlur2x2(const std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  std::vector<int> img2(img.begin(), img.end());

  // Horizontal Kernel: for each row y
  for (int y = 0; y < imgSize; ++y) {
    // Get kernel for first pixel of row y, using padding
    int value = img.at((-1*imgSize + y) % img.size()) + img.at(0*imgSize + y);
    // Division that would usually happen here is omitted
    img2[0*imgSize + y] = value;

    // Go through the rest of row y
    for (int x = 1; x < imgSize; ++x) {
      // remove the previous pixel
      value -= img.at(((x - 2)*imgSize + y) % img.size());
      // add the new pixel
      value += img.at((x*imgSize + y) % img.size());
      // save result
      img2[x*imgSize + y] = value;
    }
  }

  // Now apply the vertical kernel to img2

  // Create new output image
  std::vector<int> img3(img2.begin(), img2.end());

  // Vertical Kernel: for each column x
  for (int x = 0; x < imgSize; ++x) {
    // Get kernel for first pixel of column x with padding
    int value = img2.at((x*imgSize - 1) % img.size()) + img2.at(x*imgSize + 0);
    // Division that would usually happen here is omitted
    img3[x*imgSize + 0] = value;

    // Go through the rest of column x
    for (int y = 1; y < imgSize; ++y) {
      // remove the previous pixel
      value -= img2.at((x*imgSize + y - 2) % img.size());
      // add the new pixel
      value += img2.at((x*imgSize + y) % img.size());
      // save result
      img3[x*imgSize + y] = value;
    }
  }
  return img3;
}

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838
// use a 4x4 matrix for the tests
#define MATRIX_SIZE 4

class BoxBlurTest : public ::testing::Test {  /* NOLINT (predictable sequence expected) */
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
    /* We assume a row major layout of the matrix, where the first element is the bottom left pixel */
    for (int64_t row = size - 1; row >= 0; --row) {
      std::cout << matrix.at(0*size + row);
      for (size_t col = 1; col < size; ++col) {
        std::cout << "\t" << std::setw(4) << matrix.at(col*size + row);
      }
      std::cout << std::endl;
    }
  }
};

/// Test to ensure that naiveBoxBlur and fastBoxBlur actually compute the same thing!
TEST_F(BoxBlurTest, NaiveBoxBlur_FastBoxBlur_Equivalence) {  /* NOLINT */

  size_t size = 16;
  std::vector<int> img;
  BoxBlurTest::getInputMatrix(size, img);
  //  std::cout << "img:" << std::endl;
  //  printMatrix(size, img);

  auto naive = naiveBoxBlur(img);
  //  std::cout << "naive:" << std::endl;
  //  printMatrix(size, naive);

  auto fast = fastBoxBlur(img);
  //  std::cout << "fast:" << std::endl;
  //  printMatrix(size, fast);

  EXPECT_EQ(fast, naive);
}


/// Test to ensure that 2x2 naiveBoxBlur and fastBoxBlur actually compute the same thing!
TEST_F(BoxBlurTest, NaiveBoxBlur2x2_FastBoxBlur2x2_Equivalence) {  /* NOLINT */

  size_t size = 16;
  std::vector<int> img;
  BoxBlurTest::getInputMatrix(size, img);
  // std::cout << "img:" << std::endl;
  // printMatrix(size, img);

  auto naive = naiveBoxBlur2x2(img);
  // std::cout << "naive:" << std::endl;
  // printMatrix(size, naive);

  auto fast = fastBoxBlur2x2(img);
  // std::cout << "fast:" << std::endl;
  // printMatrix(size, fast);

  EXPECT_EQ(fast, naive);
}

#ifdef HAVE_SEAL_BFV
/// Test to ensure that 2x2 fastBoxBlur and non-batched encrypted actually compute the same thing!
TEST_F(BoxBlurTest, NonBatchedBoxBlur2x2_FastBoxBlur2x2_Equivalence) {  /* NOLINT */

  size_t size = 16;
  std::vector<int> img;
  size_t poly_modulus_degree = 2 << 12;
  BoxBlurTest::getInputMatrix(size, img);

  auto fast = fastBoxBlur2x2(img);

  MultiTimer dummy = MultiTimer();
  auto nonBatched = encryptedFastBoxBlur2x2(dummy, img, poly_modulus_degree);
  std::vector<int> enc(begin(nonBatched), end(nonBatched));

  EXPECT_EQ(fast, enc);
}

/// Test to ensure that fastBoxBlur and encryptedBoxBlur compute the same thing
TEST_F(BoxBlurTest, EncryptedBoxBlur_FastBoxBlur_Equivalence) { /* NOLINT */

  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  BoxBlurTest::getInputMatrix(size, img);

  auto fast = fastBoxBlur(img);

  auto dummy = MultiTimer();
  auto encrypted = encryptedBatchedBoxBlur(dummy, img, poly_modulus_degree);
  std::vector<int> enc(begin(encrypted), end(encrypted));
  enc.resize(fast.size()); // Is there a more efficient way to do this?

  EXPECT_EQ(fast, enc);
}

/// Test to ensure that naiveBoxBlur2x2 and encryptedBatchedBoxBlur_Porcupine compute the same thing
TEST_F(BoxBlurTest, Porcupine_Naive_Equivalence) { /* NOLINT */

  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  BoxBlurTest::getInputMatrix(size, img);

  auto naive = naiveBoxBlur2x2(img);

  auto dummy = MultiTimer();
  auto encrypted = encryptedBatchedBoxBlur_Porcupine(dummy, img, poly_modulus_degree);
  std::vector<int> enc(begin(encrypted), end(encrypted));
  enc.resize(naive.size()); // Is there a more efficient way to do this?

  EXPECT_EQ(naive, enc);
}
#endif