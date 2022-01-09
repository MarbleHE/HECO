#include <set>
#include <random>
#include "gtest/gtest.h"
#include "abc/ast/AbstractNode.h"
#include "abc/ast_parser/Parser.h"
#include "abc/ast/Assignment.h"
#include "abc/ast/Variable.h"

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838
// use a 4x4 matrix for the tests
#define MATRIX_SIZE 4

/// Original, plain C++ program for LaplacianSharpening
/// This uses a 3x3 Kernel and applies it by sliding across the 2D image
///             | 1  1  1 |
///   w = -1 *  | 1 -8  1 |
///             | 1  1  1 |
/// No padding is used, so the image is 2px smaller in each dimension
/// The filtered image is then added to the original image at 50% intensity.
///
/// This program is inspired by RAMPARTS, presented by Archer et al. in 2019
/// The only modification that we do is instead of computing
///     img2[x][y] = img[x][y] - (value/2),
/// we compute
///     img2[x][y] = 2*img[x][y] - value
/// to avoid division which is unsupported in BFV
/// The client can easily divide the result by two after decryption
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> laplacianSharpening(const std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  std::vector<std::vector<int>> weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  std::vector<int> img2(img.begin(), img.end());
  for (int x = 1; x < imgSize - 1; ++x) {
    for (int y = 1; y < imgSize - 1; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
          value = value + weightMatrix.at(i + 1).at(j + 1)*img.at((x + i)*imgSize + (y + j));
        }
      }
      img2[imgSize*x + y] = 2*img.at(x*imgSize + y) - value;
    }
  }
  return img2;
}

class KernelTest : public ::testing::Test {  /* NOLINT (predictable sequence expected) */
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

  /// Creates a secret int image vector of the given size and an int imgSize that stores the size
  std::unique_ptr<AbstractNode> getInputs(size_t size) {
    // we use the getInputMatrix method here to create the inputs for the input AST in a reproducible and flexible way
    std::vector<int> inputValues;
    getInputMatrix(size, inputValues);

    std::stringstream inputString;
    inputString << "secret int image = { ";
    std::copy(inputValues.begin(), inputValues.end() - 1, std::ostream_iterator<int>(inputString, ", "));
    inputString << inputValues.back(); // add the last element with no delimiter
    inputString << " };" << std::endl;
    inputString << "int imgSize = " << size << ";" << std::endl;

    return Parser::parse(inputString.str());
  }

  static std::unique_ptr<AbstractNode> getOutputs() {
    const char *outputs = R""""(
      resultImage = img2;
    )"""";
    return Parser::parse(std::string(outputs));
  }

  static std::unique_ptr<AbstractNode> getLaplaceSharpeningProgram() {
    std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
    return getLaplaceSharpeningProgramAndNodes(createdNodes);
  }

  static std::unique_ptr<AbstractNode> getLaplaceSharpeningProgramAndNodes(std::vector<std::reference_wrapper<
      AbstractNode>> &createdNodes) {
    // program's input
    const char *inputs = R""""(
      public void laplacianSharpening(int imageVec) {
        int weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
        secret int img2 = image;
        for (int x = 1; x < imgSize - 1; x = x + 1) {
          for (int y = 1; y < imgSize - 1; y = y + 1) {
            secret int value = 0;
            for (int j = -1; j < 2; j = j + 1) {
              for (int i = -1; i < 2; i = i + 1) {
                value = value + weightMatrix[i+1][j+1] * image[(x+i)*imgSize+(y+j)];
              }
            }
            img2[x*imgSize+y] = 2 * image[x*imgSize+y] - value;
          }
        }
        return;  // img2 contains result
      }
    )"""";
    return Parser::parse(std::string(inputs), createdNodes);
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
};