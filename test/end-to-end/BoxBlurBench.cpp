#ifdef HAVE_SEAL_BFV

#include <complex>
#include <random>

#include "BoxBlurTest.h"
#include "MultiTimer.h"

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838

// Number of iterations for the benchmark
#define INTER_COUNT 5

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
  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  getInputMatrix(size, img);

  MultiTimer expertTimer = MultiTimer();
  for (int i = 0; i < INTER_COUNT; ++i) {
    auto encrypted = encryptedBatchedBoxBlur(expertTimer, img, poly_modulus_degree);
    expertTimer.addIteration();
  }

  MultiTimer porcupineTimer = MultiTimer();
  for (int i = 0; i < INTER_COUNT; ++i) {
    auto encrypted = encryptedBatchedBoxBlur_Porcupine(porcupineTimer, img, poly_modulus_degree);
    porcupineTimer.addIteration();
  }

  MultiTimer naiveTimer = MultiTimer();
  for (int i = 0; i < INTER_COUNT; ++i) {
    auto encrypted = encryptedFastBoxBlur2x2(naiveTimer, img, poly_modulus_degree);
    naiveTimer.addIteration();
  }

  std::cout << "naive:" << std::endl;
  naiveTimer.printToStream(std::cout);
  naiveTimer.printToFile("bb_naive_8192_4096.csv");

  std::cout << "expert:" << std::endl;
  expertTimer.printToStream(std::cout);
  expertTimer.printToFile("bb_expert_8192_4096.csv");

  std::cout << "porcupine:" << std::endl;
  porcupineTimer.printToStream(std::cout);
  porcupineTimer.printToFile("bb_porcupine_8192_4096.csv");
}

#endif
