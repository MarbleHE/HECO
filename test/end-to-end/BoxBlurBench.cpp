#ifdef HAVE_SEAL_BFV

#include <complex>
#include <random>
#include <string>

#include "BoxBlurTest.h"
#include "MultiTimer.h"

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838

// Number of iterations for the benchmark
#define INTER_COUNT 5

#define BENCH_FUNCTION(VERSION, FUNCTION_NAME) if (argv[1] == std::string(#VERSION)) { \
  for (int i = 0; i < INTER_COUNT; ++i) {                                              \
    auto result = FUNCTION_NAME(timer, img, poly_modulus_degree);                      \
    timer.addIteration();                                                              \
  }                                                                                    \
  timer.printToFile("BoxBlur_" #VERSION "_64.csv");                                  \
}

void getInputMatrix(size_t size, std::vector<std::vector<int>> &destination) {
  // reset the RNG to make sure that every call to this method results in the same numbers
  auto randomEngine = std::default_random_engine(RAND_SEED);
  auto myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);

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


int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: bb-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          -naive" << std::endl;
    std::cout << "          -expert" << std::endl;
    std::cout << "          -porcupine" << std::endl;
    std::exit(1);
  }

  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  getInputMatrix(size, img);

  MultiTimer timer = MultiTimer();
  BENCH_FUNCTION(naive, encryptedFastBoxBlur2x2);
  BENCH_FUNCTION(expert, encryptedBatchedBoxBlur);
  BENCH_FUNCTION(porcupine, encryptedBatchedBoxBlur_Porcupine);

  return 0;
}

#endif
