#include "main.h"
#include "examples/genAstDemo.h"
#include "ast_opt/ast/OpSymbEnum.h"
#include "ast_opt/evaluation/EvaluationAlgorithms.h"

int main() {
  // test linear regression
  // expected: a = -1.13793 and b = 2.89655
  std::vector<std::pair<float, float>> input = {{0, -1}, {2, 5}, {5, 12}, {7, 20}};
  auto[a, b] = EvaluationAlgorithms::runLinearRegression(input);
  std::cout << "a = " << a << ", b = " << b << std::endl;

  // test polynomial regression
  // expected: y = 1 + 2x + 3x^2
  std::vector<int> x(11);
  iota(x.begin(), x.end(), 0);
  std::vector<int> y{1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321};
  EvaluationAlgorithms::runPolynomialRegression(x, y);

  // test multivariate regression


  // test sobel filter detection
  std::vector<int> img1{
      {01, 02, 03, 04, 05, 06, 07, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36,
       37, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 61, 62, 63, 64, 65, 66, 67},
  };
  EvaluationAlgorithms::runSobelFilter(img1);

  // test 8-neighbour laplacian sharpening filter
  std::vector<std::vector<int>> img2{
      {01, 02, 03, 04, 05, 06, 07},
      {11, 12, 13, 14, 15, 16, 17},
      {21, 22, 23, 24, 25, 26, 27},
      {31, 32, 33, 34, 35, 36, 37},
      {41, 42, 43, 44, 45, 46, 47},
      {51, 52, 53, 54, 55, 56, 57},
      {61, 62, 63, 64, 65, 66, 67},
  };
  EvaluationAlgorithms::runLaplacianSharpeningAlgorithm(img2);

  // test harris corner/edge detection

}


