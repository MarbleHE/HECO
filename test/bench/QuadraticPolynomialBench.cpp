#include <complex>

#include "QuadraticPolynomial.h"
#include "BenchmarkHelper.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "USAGE: qp-bench [version]" << std::endl;
    std::cout << "       versions:" << std::endl;
    std::cout << "          - Naive" << std::endl;
    std::cout << "          - HECO" << std::endl;
    std::cout << "          - Porcupine" << std::endl;
    std::exit(1);
  }

  size_t poly_modulus_degree = 2 << 13;
  size_t vec_size = 64;
  std::vector<int> x(vec_size);
  std::vector<int> y(vec_size);
  std::vector<int> a(vec_size);
  std::vector<int> b(vec_size);
  std::vector<int> c(vec_size);

  getRandomVector(x);
  getRandomVector(y);
  getRandomVector(a);
  getRandomVector(b);
  getRandomVector(c);

  BENCH_FUNCTION(QuadraticPolynomial, Naive, encryptedQuadraticPolynomialNaive, a, b, c, x, y);
  BENCH_FUNCTION(QuadraticPolynomial, HECO, encryptedQuadraticPolynomialBatched, a, b, c, x, y);
  BENCH_FUNCTION(QuadraticPolynomial, Porcupine, encryptedQuadraticPolynomialPorcupine, a, b, c, x, y);

  return 0;
}