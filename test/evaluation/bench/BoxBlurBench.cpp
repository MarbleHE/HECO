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

  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  if (argc == 3) {
    size = atoi(argv[2]);
    size_t s_square = size*size;
    // Ensure that total items is less than half of modulus
    // Search for poly_modulus degree that's a power of 2 and large enough
    while (s_square > poly_modulus_degree / 2) poly_modulus_degree *= 2;
  }

  std::vector<int> img;
  getInputMatrix(size, img);

  BENCH_FUNCTION(BoxBlur, Naive, encryptedFastBoxBlur2x2, img);
  BENCH_FUNCTION(BoxBlur, HECO, encryptedBatchedBoxBlur, img);
  BENCH_FUNCTION(BoxBlur, Porcupine, encryptedBatchedBoxBlur_Porcupine, img);

  return 0;
}
