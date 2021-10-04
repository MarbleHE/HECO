#include <cmath>
#include <random>
#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/utilities/Scope.h"
#include "ast_opt/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/parser/Parser.h"
#include "gtest/gtest.h"

#include "BoxBlurTest.h"

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

/// Encrypted BoxBlur, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedBatchedBoxBlur(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights)
{
  int t0 = timer.startTimer(); // keygen timer start
  // Input Check
  if (img.size() != poly_modulus_degree / 2)
  {
    std::cerr << "WARNING: BatchedBoxBlur might be incorrect when img.size() does not match N/2." << std::endl;
  }

  /// Rotations for 3x3 Kernel
  /// Offsets correspond to the different kernel positions
  int img_size = (int)std::sqrt(img.size());
  std::vector<int> rotations = { -img_size + 1, 1,  img_size + 1, -img_size, 0, img_size,
                                 -img_size - 1, -1, img_size - 1 };
  // Context Setup
  // std::cout << "Setting up SEAL Context" << std::endl;
  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 30));
  seal::SEALContext context(parameters);

  /// Create keys
  // std::cout << "Generating Keys & Helper Objects" << std::endl;
  seal::KeyGenerator keygen(context);
  seal::SecretKey secretKey = keygen.secret_key();
  seal::PublicKey publicKey;
  keygen.create_public_key(publicKey);
  seal::GaloisKeys galoisKeys;
  keygen.create_galois_keys(rotations, galoisKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context); // changed from this: EVALUATOR evaluator(context);

  // Create Weight Matrix
  std::vector<int> weight_matrix = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  timer.stopTimer(t0); // keygen timer stop

  auto encryptTime = timer.startTimer(); // encryption timer start
  // Encode & Encrypt the image
  // std::cout << "Encoding & Encrypting Image" << std::endl;
  seal::Plaintext img_ptxt;
  seal::Ciphertext img_ctxt;
  std::vector<uint64_t> long_vec = std::vector<uint64_t>(img.begin(), img.end());
  // std::cout << long_vec.size() << std::endl;
  encoder.encode(long_vec, img_ptxt);
  encryptor.encrypt(img_ptxt, img_ctxt);

  // Encode & (if needed) Encrypt the weights
  std::vector<seal::Plaintext> w_ptxts(weight_matrix.size());
  std::vector<seal::Ciphertext> w_ctxts(weight_matrix.size());
  for (size_t i = 0; i < weight_matrix.size(); ++i)
  {
    encoder.encode(std::vector<int64_t>(encoder.slot_count(), weight_matrix[i]), w_ptxts[i]);
    if (encrypt_weights)
    {
      encryptor.encrypt(w_ptxts[i], w_ctxts[i]);
    }
  }
  timer.stopTimer(encryptTime); // encryption timer stop

  int t1 = timer.startTimer(); // computation timer start
  // Create rotated copies of the image and multiply by weights
  // std::cout << "Applying Kernel" << std::endl;
  std::vector<seal::Ciphertext> rotated_img_ctxts(9, seal::Ciphertext(context));
  for (size_t i = 0; i < rotations.size(); ++i)
  {
    evaluator.rotate_rows(img_ctxt, rotations[i], galoisKeys, rotated_img_ctxts[i]);

    if (encrypt_weights)
    {
      evaluator.multiply_inplace(rotated_img_ctxts[i], w_ctxts[i]);
      // relinearization not needed since no more mults coming up
    }
    else
    {
      // If the weight is ptxt and one, we can skip this entirely
      if (weight_matrix[i] != 1)
      {
        evaluator.multiply_plain_inplace(rotated_img_ctxts[i], w_ptxts[i]);
      }
    }
  }

  // Sum up all the ciphertexts
  seal::Ciphertext result_ctxt(context);
  evaluator.add_many(rotated_img_ctxts, result_ctxt);
  timer.stopTimer(t1); // computation timer stop

  int t2 = timer.startTimer(); // decrypt timer start
  // Decrypt & Return result
  // std::cout << "Decrypting Result" << std::endl;
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(t2); // decrypt timer stop
  // std::cout << result.size() << std::endl;
  return result;
}

//TODO: Need 2x2 expert boxblur example

/// Encrypted BoxBlur, using the pseudocode given by the porcupine paper:
///  Ciphertext boxblur(Ciphertext c0, int h, int w)
///      Ciphertext c1 = rotate(c0, -1 * w)
///      Ciphertext c2 = add(c0, c1)
///      Ciphertext c3 = rotate(c2, -1)
///      return add(c2, c3)
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \return transformed image
std::vector<int64_t> encryptedBatchedBoxBlur_Porcupine(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree)
{
  /* Setup */
  auto t0 = timer.startTimer(); // keygen timer start

  seal::EncryptionParameters params(seal::scheme_type::bfv);

  params.set_poly_modulus_degree(poly_modulus_degree);
  params.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
  params.set_plain_modulus(seal::PlainModulus::Batching(poly_modulus_degree, 20));
  seal::SEALContext context(params);

  // Create keys objects
  seal::KeyGenerator keygen(context);
  seal::SecretKey secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);
  seal::GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_keys);

  // Create helper objects
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batch_encoder(context);
  seal::Encryptor encryptor(context, public_key);
  seal::Decryptor decryptor(context, secret_key);
  timer.stopTimer(t0); // keygen timer stop

  auto encTime = timer.startTimer(); // encryption timer start
  seal::Plaintext plain;
  std::vector<uint64_t> long_vec = std::vector<uint64_t>(img.begin(), img.end());
  batch_encoder.encode(long_vec, plain);

  seal::Ciphertext c0;
  encryptor.encrypt(plain, c0);
  timer.stopTimer(encTime); // encryption timer stop

  /* Computation */
  auto t1 = timer.startTimer();

  // Ciphertext c1 = rotate(c0, -1 * w)
  seal::Ciphertext c1;
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  evaluator.rotate_rows(c0, -1 * imgSize, galois_keys, c1);

  // Ciphertext c2 = add(c0, c1)
  seal::Ciphertext c2;
  evaluator.add(c0, c1, c2);

  // Ciphertext c3 = rotate(c2, -1)
  seal::Ciphertext c3;
  evaluator.rotate_rows(c2, -1, galois_keys, c3);

  // return add(c2, c3)
  seal::Ciphertext result;
  evaluator.add(c2, c3, result);
  timer.stopTimer(t1);

  /* Decrypt */
  auto t2 = timer.startTimer();

  seal::Plaintext decrypted;
  decryptor.decrypt(result, decrypted);

  std::vector<int64_t> retVal;
  batch_encoder.decode(decrypted, retVal);
  timer.stopTimer(t2);
  return retVal;
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
        std::cout << "\t" << matrix.at(col*size + row);
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

TEST_F(BoxBlurTest, clearTextEvaluationNaive) { /* NOLINT */
  /// program's input
  const char *inputs = R""""(
      int img = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      int imgSize = 4;
    )"""";
  auto astInput = Parser::parse(std::string(inputs));


  /// program specification
  /// TODO: Doesn't use wrap-around padding since the modulo returns negative numbers here :(
  const char *program = R""""(
    int weightMatrix = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int img2 = img;
    for (int x = 1; x < imgSize-1; x = x + 1) {
      for (int y = 1; y < imgSize-1; y = y + 1) {
        int value = 0;
        for (int j = -1; j < 2; j = j + 1) {
          for (int i = -1; i < 2; i = i + 1) {
            value = value + weightMatrix[(i + 1)*3 +j + 1]
                *img[((x + i)*imgSize + (y + j))];
          }
        }
        img2[imgSize*x + y] = value;
      }
    }
    return img2;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      img2 = img2;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  auto scf = std::make_unique<DummyCiphertextFactory>();
  auto tcv = std::make_unique<TypeCheckingVisitor>();

  // create and prepopulate TypeCheckingVisitor
  auto registerInputVariable = [&tcv](Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  };

  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "img", Datatype(Type::INT, false));
  registerInputVariable(*rootScope, "imgSize", Datatype(Type::INT, false));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  //TODO: Change it so that by passing in an empty secretTaintingMap, we can get the RuntimeVisitor to execute everything "in the clear"!
  auto empty = std::unordered_map<std::string, bool>();
  RuntimeVisitor srv(*scf, *astInput, empty);
  srv.executeAst(*astProgram);


  /// A helper method that takes the result produced by the RuntimeVisitor (result) and a list of expected
  /// (identifier, vector of values) pairs that the program should have returned.
  /// \param result The generated result retrieved by getOutput from the RuntimeVisitor.
  /// \param expectedResult The expected result that the program should have been produced.
  auto assertResult = [&scf](const OutputIdentifierValuePairs &result,
                             const std::unordered_map<std::string, std::vector<int64_t>> &expectedResult) {
    // Check that the number of results match the number of expected results
    EXPECT_EQ(result.size(), expectedResult.size());

    for (const auto &[identifier, cipherClearText] : result) {
      // Check that the result we are currently processing is indeed an expected result
      EXPECT_EQ(expectedResult.count(identifier), 1);

      // for checking the value, distinguish between a ciphertext (requires decryption) and plaintext
      std::vector<int64_t> plainValues;
      if (auto ciphertext = dynamic_cast<AbstractCiphertext *>(cipherClearText.get())) {        // result is a ciphertxt
        scf->decryptCiphertext(*ciphertext, plainValues);
        const auto &expResultVec = expectedResult.at(identifier);
        // to avoid comparing the expanded values (last element of ciphertext is repeated to all remaining slots), we
        // only compare the values provided in the expectedResult map
        for (int i = 0; i < expResultVec.size(); ++i) {
          EXPECT_EQ(plainValues.at(i), expectedResult.at(identifier).at(i));
        }
      } else if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(cipherClearText.get())) {   // result is a cleartext
        auto cleartextData = cleartextInt->getData();
        // required to convert vector<int> to vector<int64_t>
        plainValues.insert(plainValues.end(), cleartextData.begin(), cleartextData.end());
        EXPECT_EQ(plainValues, expectedResult.at(identifier));
      } else if (auto
          cleartextBool = dynamic_cast<Cleartext<bool> *>(cipherClearText.get())) {   // result is a cleartext
        auto cleartextData = cleartextBool->getData();
        // required to convert vector<int> to vector<int64_t>
        plainValues.insert(plainValues.end(), cleartextData.begin(), cleartextData.end());
        EXPECT_EQ(plainValues, expectedResult.at(identifier));
      } else {
        throw std::runtime_error("Could not determine type of result.");
      }
    }
  };

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["img2"] = {1, 1, 1, 1, 1, 9, 9, 1, 1, 9, 9, 1, 1, 1, 1, 1};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}