#include <cmath>
#include <random>
#include "gtest/gtest.h"

#include "RobertsCrossTest.h"

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

#ifdef HAVE_SEAL_BFV
/// Output is squared to elide square root
/// Ciphertext roberts_cross(Ciphertext c0, int h, int w)
///     Ciphertext c1 = rotate(c0, w)
///     Ciphertext c2 = rotate(c0, 1)
///     Ciphertext c3 = sub(c1, c2)
///     Ciphertext c4 = square(c4)
///     c4 = relinearize(c4)
///     Ciphertext c5 = rotate(c0, w + 1)
///     Ciphertext c6 = sub(c5, c0)
///     Ciphertext c7 = square(c6)
///     c7 = relinearize(c7)
///     return add(c4, c7)
std::vector<int64_t> encryptedRobertsCrossPorcupine(MultiTimer &timer, std::vector<int> &img, size_t poly_modulus_degree)
{
  int img_size = (int)std::sqrt(img.size());

  // Context Setup
  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 30));
  seal::SEALContext context(parameters);

  /// Create keys
  seal::KeyGenerator keygen(context);
  seal::SecretKey secretKey = keygen.secret_key();
  seal::PublicKey publicKey;
  keygen.create_public_key(publicKey);
  seal::GaloisKeys galoisKeys;
  keygen.create_galois_keys(galoisKeys);
  seal::RelinKeys relinKeys;
  keygen.create_relin_keys(relinKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);

  // Encode & Encrypt the image
  seal::Plaintext img_ptxt;
  seal::Ciphertext img_ctxt;
  encoder.encode(std::vector<uint64_t>(img.begin(), img.end()), img_ptxt);
  encryptor.encrypt(img_ptxt, img_ctxt);

  // Ciphertext c1 = rotate(c0, w)
  seal::Ciphertext c1;
  evaluator.rotate_rows(img_ctxt, img_size, galoisKeys, c1);
  // Ciphertext c2 = rotate(c0, 1)
  seal::Ciphertext c2;
  evaluator.rotate_rows(img_ctxt, 1, galoisKeys, c2);
  // Ciphertext c3 = sub(c1, c2)
  seal::Ciphertext c3;
  evaluator.sub(c1, c2, c3);
  // Ciphertext c4 = square(c4) //TODO: There is an error here
  seal::Ciphertext c4;
  evaluator.square(c3, c4);
  // c4 = relinearize(c4)
  evaluator.relinearize_inplace(c4, relinKeys);
  // Ciphertext c5 = rotate(c0, w + 1)
  seal::Ciphertext c5;
  evaluator.rotate_rows(img_ctxt, img_size + 1, galoisKeys, c5);
  // Ciphertext c6 = sub(c5, c0)
  seal::Ciphertext c6;
  evaluator.sub(c5, img_ctxt, c6);
  // Ciphertext c7 = square(c6)
  seal::Ciphertext c7;
  evaluator.square(c6, c7);
  // c7 = relinearize(c7)
  evaluator.relinearize_inplace(c7, relinKeys);
  // return add(c4, c7)
  seal::Ciphertext result_ctxt;
  evaluator.add(c4, c7, result_ctxt);

  // Decrypt & Return result
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  return result;
}
#endif //HAVE_SEAL_BFV


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
      std::cout << matrix.at(0*size + row);
      for (size_t col = 1; col < size; ++col) {
        std::cout << "\t" << matrix.at(col*size + row);
      }
      std::cout << std::endl;
    }
  }
};

#ifdef HAVE_SEAL_BFV
TEST_F(RobertsCrossKernelTest, Clear_EncryptedPorcupine_Equivalence) {
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