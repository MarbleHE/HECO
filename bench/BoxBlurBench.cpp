#include <complex>

#include "BoxBlur.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: bb-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          - Naive" << std::endl;
    std::cout << "          - HECO" << std::endl;
    std::cout << "          - Porcupine" << std::endl;
    std::exit(1);
  }

  size_t size = 64;
  if (argc == 3) {
    size = atoi(argv[2]);
  }

  size_t poly_modulus_degree = size * size * 2;
  if (poly_modulus_degree < 8192) {
    poly_modulus_degree = 8192;
  }
  std::vector<int> img;
  getInputMatrix(size, img);

  BENCH_FUNCTION(BoxBlur, Naive, encryptedFastBoxBlur2x2, img);
  BENCH_FUNCTION(BoxBlur, HECO, encryptedBatchedBoxBlur, img);
  BENCH_FUNCTION(BoxBlur, Porcupine, encryptedBatchedBoxBlur_Porcupine, img);

  return 0;
}
