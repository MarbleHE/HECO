#include <complex>

#include "HammingDistance.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: ham-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          -porcupine" << std::endl;
    std::cout << "          -expert" << std::endl;
    std::cout << "          -naive" << std::endl;
    std::exit(1);
  }

  // Create two vectors of bits (booleans),
  // TODO: Create test values from fixed random seed
  std::vector<bool> a(4, 0);
  std::vector<bool> b(4, 1);
  size_t poly_modulus_degree = 2 << 12;

  BENCH_FUNCTION(HammingDistance, porcupine, encryptedHammingDistancePorcupine, a, b);
  BENCH_FUNCTION(HammingDistance, expert, encryptedBatchedHammingDistance, a, b);
  BENCH_FUNCTION(HammingDistance, naive, encryptedNaiveHammingDistance, a, b);

  return 0;
}
