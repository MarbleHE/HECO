#include <complex>

#include "L2Distance.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: l2-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          - Porcupine" << std::endl;
    std::cout << "          - HECO" << std::endl;
    std::cout << "          - Naive" << std::endl;

    std::exit(1);
  }
  size_t vec_size = 4;
  if (argc == 3) {
    vec_size = atoi(argv[2]);
  }

  // Create two vectors of bits (booleans),
  // TODO: Create test values from fixed random seed
  std::vector<int> a(vec_size, 10);
  std::vector<int> b(vec_size, 20);
  size_t poly_modulus_degree = 2 << 12;

  BENCH_FUNCTION(L2Distance, Porcupine, encryptedL2DistanceSquared_Porcupine, a, b);
  BENCH_FUNCTION(L2Distance, HECO, encryptedBatchedSquaredL2Distance, a, b);
  BENCH_FUNCTION(L2Distance, Naive, encryptedL2DistanceSquared_Naive, a, b);

  return 0;
}