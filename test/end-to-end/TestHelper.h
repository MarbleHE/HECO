#ifndef AST_OPTIMIZER_TESTHELPER_H
#define AST_OPTIMIZER_TESTHELPER_H

#include <random>

// use this fixed seed to enable reproducibility of the inputs
#define RAND_SEED 4673838

void getRandomVector(std::vector<int> &destination) {
  // reset the RNG to make sure that every call to this method results in the same numbers
  auto randomEngine = std::default_random_engine(RAND_SEED);
  auto myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);

  for (int & i : destination) {
    i = myUnifIntDist(randomEngine);
  }
}

void getRandomVector(std::vector<int> &destination, int nonce) {
  // reset the RNG to make sure that every call to this method results in the same numbers
  auto randomEngine = std::default_random_engine(RAND_SEED + nonce);
  auto myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);

  for (int & i : destination) {
    i = myUnifIntDist(randomEngine);
  }
}

void printVector(std::vector<int> &vector) {
  for (int value : vector) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
}

#endif