#include <complex>

#include "HammingDistance.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: ham-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          - Porcupine" << std::endl;
    std::cout << "          - HECO" << std::endl;
    std::cout << "          - Naive" << std::endl;

    std::exit(1);
  }

  // Create two vectors of bits (booleans),
  // TODO: Create test values from fixed random seed
  std::vector<bool> a(4, 0);
  std::vector<bool> b(4, 1);
  size_t poly_modulus_degree = 2 << 12;

  BENCH_FUNCTION(HammingDistance, Porcupine, encryptedHammingDistancePorcupine, a, b);
  BENCH_FUNCTION(HammingDistance, HECO, encryptedBatchedHammingDistance, a, b);
  BENCH_FUNCTION(HammingDistance, Naive, encryptedNaiveHammingDistance, a, b);

  return 0;
}
