#include <complex>

#include "GxKernel.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: gx-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          -porcupine" << std::endl;
    std::exit(1);
  }

  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  getInputMatrix(size, img);

  BENCH_FUNCTION(GxKernel, porcupine, encryptedBatchedGxKernelPorcupine, img);
  BENCH_FUNCTION(GxKernel, expert, encryptedBatchedGxKernel, img);

  return 0;
}