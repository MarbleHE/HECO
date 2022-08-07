#include <complex>

#include "LinearPolynomial.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: lp-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          - Naive" << std::endl;
    std::cout << "          - Porcupine" << std::endl;
    std::cout << "          - HECO" << std::endl;
    std::exit(1);
  }

  size_t poly_modulus_degree = 2 << 12;
  size_t vec_size = std::sqrt(poly_modulus_degree / 2);
  std::vector<int> x(vec_size);
  std::vector<int> y(vec_size);
  std::vector<int> a(vec_size);
  std::vector<int> b(vec_size);

  getRandomVector(x);
  getRandomVector(y);
  getRandomVector(a);
  getRandomVector(b);

  BENCH_FUNCTION(LinearPolynomial, Naive, encryptedLinearPolynomialNaive, a, b, x, y);
  BENCH_FUNCTION(LinearPolynomial, HECO, encryptedLinearPolynomialBatched, a, b, x, y);
  BENCH_FUNCTION(LinearPolynomial, Porcupine, encryptedLinearPolynomialPorcupine, a, b, x, y);

  return 0;
}
