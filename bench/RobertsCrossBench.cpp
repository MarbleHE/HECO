#include <complex>

#include "RobertsCross.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: rc-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          -porcupine" << std::endl;
    std::exit(1);
  }

  size_t poly_modulus_degree = 2 << 12;
  size_t size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> img;
  getInputMatrix(size, img);

  BENCH_FUNCTION(RobertsCross, porcupine, encryptedRobertsCrossPorcupine, img);

  return 0;
}