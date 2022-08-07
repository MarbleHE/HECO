#include "LaplaceSharpening.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: ls-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          - NaiveBatched" << std::endl;
    std::cout << "          - HECO" << std::endl;
    std::cout << "          - Naive" << std::endl;

    std::exit(1);
  }

  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  getInputMatrix(size, img);

  BENCH_FUNCTION(LaplaceSharpening, NaiveBatched, encryptedNaiveBatchedLaplacianSharpening, img);
  BENCH_FUNCTION(LaplaceSharpening, HECO, encryptedLaplacianSharpening, img);
  BENCH_FUNCTION(LaplaceSharpening, Naive, encryptedNaiveLaplaceSharpening, img);

  return 0;
}
