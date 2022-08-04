#ifndef AST_OPTIMIZER_BENCHMARKHELPER_H
#define AST_OPTIMIZER_BENCHMARKHELPER_H

#include <random>
#include <string>

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838

// Number of iterations for the benchmark
#define ITER_COUNT 10

#define BENCH_FUNCTION(PROBLEM, VERSION, FUNCTION_NAME, ...) if (argv[1] == std::string(#VERSION)) { \
  MultiTimer timer = MultiTimer();                                                                   \
  for (int i = 0; i < ITER_COUNT; ++i) {                                                             \
    FUNCTION_NAME(timer, __VA_ARGS__, poly_modulus_degree);                                          \
    timer.addIteration();                                                                            \
  }                                                                                                  \
  std::string size_string = "64";                                                                    \
  if (argc == 3) {                                                                                   \
    size_string = argv[2];                                                                           \
  }                                                                                                  \
  timer.printToFile( #PROBLEM "_" #VERSION "_"+size_string+".csv");                                  \
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

void getRandomVector(std::vector<int> &destination) {
  // reset the RNG to make sure that every call to this method results in the same numbers
  auto randomEngine = std::default_random_engine(RAND_SEED);
  auto myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);

  for (int & i : destination) {
    i = myUnifIntDist(randomEngine);
  }
}

#endif//AST_OPTIMIZER_BENCHMARKHELPER_H
