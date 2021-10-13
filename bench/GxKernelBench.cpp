#include <complex>

#include "GxKernel.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: gx-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          - Porcupine" << std::endl;
    std::cout << "          - HECO" << std::endl;
    std::exit(1);
  }

  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  getInputMatrix(size, img);

  BENCH_FUNCTION(GxKernel, Porcupine, encryptedBatchedGxKernelPorcupine, img);
  BENCH_FUNCTION(GxKernel, HECO, encryptedBatchedGxKernel, img);

  return 0;
}
