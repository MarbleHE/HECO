#include "EvaluationAlgorithms.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <functional>

typedef std::vector<std::vector<int>> VecInt2D;

// Credits to Hariom Singh from stackoverflow.com (https://stackoverflow.com/a/45577531/3017719).
std::vector<int> cyclicRotate(std::vector<int> &A, int rotationFactor) {
  if (A.empty() || A.size()==1) return A;
  rotationFactor = rotationFactor%A.size();
  if (rotationFactor==0) return A;
  std::rotate(A.begin(), A.begin() + rotationFactor, A.end());
  return A;
}

std::pair<float, float> EvaluationAlgorithms::runLinearRegression(std::vector<std::pair<float, float>> datapoints) {
  int numDatapoints = datapoints.size();
  float sumX = 0, sumXX = 0, sumY = 0, sumXY = 0;
  for (int i = 0; i < numDatapoints; i++) {
    sumX = sumX + datapoints.at(i).first;
    sumXX = sumXX + datapoints.at(i).first*datapoints.at(i).first;
    sumY = sumY + datapoints.at(i).second;
    sumXY = sumXY + datapoints.at(i).first*datapoints.at(i).second;
  }

  // calculate regression parameters a, b
  float b = (static_cast<float>(numDatapoints)*sumXY - sumX*sumY)/(static_cast<float>(numDatapoints)*sumXX - sumX*sumX);
  float a = (sumY - b*sumX)/static_cast<float>(numDatapoints);

  // display result and equation of regression line (y = ax + bx)
//  std::cout << "[Result] a: " << a << ", b: " << b << std::endl;
//  std::cout << "Equation of best fit: y = " << a << " + " << b << "x";

  return std::make_pair(a, b);
}

void EvaluationAlgorithms::runPolynomialRegression(const std::vector<int> &x, const std::vector<int> &y) {
  std::vector<int> r(x.size(), 0);
  // fill r with sequentially increasing numbers 0, 1, 2, ..., x.size()
  std::iota(r.begin(), r.end(), 0);
  double meanX = std::accumulate(x.begin(), x.end(), 0.0)/x.size();
  double meanY = std::accumulate(y.begin(), y.end(), 0.0)/y.size();
  double x2m = 0.0, x3m = 0.0, x4m = 0.0;
  for (int i = 0; i < r.size(); ++i) {
    x2m += r.at(i)*r.at(i);
    x3m += r.at(i)*r.at(i)*r.at(i);
    x4m += r.at(i)*r.at(i)*r.at(i)*r.at(i);
  }
  // x2m = (∑ x_i * x_i) / N
  x2m /= r.size();
  // x3m = (∑ x_i * x_i * x_i) / N
  x3m /= r.size();
  // x4m = (∑ x_i * x_i * x_i * x_i) / N
  x4m /= r.size();

  // computes ( 0.0 + (x_1*y_1) + (x_2*y_2) + ... + (x_N + y_N) ) / N
  double xym
      = std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<double>{}, std::multiplies<double>{});
  xym /= x.size() < y.size() ? x.size() : y.size();

  double x2ym = std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0,
                                      std::plus<double>{}, [](double a, double b) { return a*a*b; });
  x2ym /= x.size() < y.size() ? x.size() : y.size();

  // compute parameters a,b,c of trend line y = a + bx + cx^{2}
  double sxx = x2m - meanX*meanX;
  double sxy = xym - meanX*meanY;
  double sxx2 = x3m - meanX*x2m;
  double sx2x2 = x4m - x2m*x2m;
  double sx2y = x2ym - x2m*meanY;
  double b = (sxy*sx2x2 - sx2y*sxx2)/(sxx*sx2x2 - sxx2*sxx2);
  double c = (sx2y*sxx - sxy*sxx2)/(sxx*sx2x2 - sxx2*sxx2);
  double a = meanY - b*meanX - c*x2m;

  // prints the computed result
  auto abc = [a, b, c](int xx) { return a + b*xx + c*xx*xx; };
//  std::cout << "y = " << a << " + " << b << "x + " << c << "x^2" << std::endl;
//  std::cout << " Input  Approximation" << std::endl;
//  std::cout << " x   y     y1" << std::endl;
  auto xit = x.cbegin();
  auto yit = y.cbegin();
  while (xit!=x.cend() && yit!=y.cend()) {
    printf("%2d %3d  %5.1f\n", *xit, *yit, abc(*xit));
    xit = std::next(xit);
    yit = std::next(yit);
  }
}

VecInt2D EvaluationAlgorithms::runLaplacianSharpeningAlgorithm(VecInt2D img) {
  VecInt2D weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  VecInt2D img2(img);
  for (int x = 1; x < img.size() - 1; ++x) {
    for (int y = 1; y < img.at(x).size() - 1; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
//          std::cout << x << ", " << y << ", " << i << ", " << j << std::endl;
          value = value + weightMatrix.at(i + 1).at(j + 1)*img.at(x + i).at(y + j);
        }
      }
      img2[x][y] = img.at(x).at(y) - (value/2);
    }
  }
  return img2;
}

std::vector<int> EvaluationAlgorithms::runSobelFilter(const std::vector<int> &img) {
  // ATTENTION: This algorithm works but has not been tested with real values yet!

  // a 3rd-degree polynomial approximation of the square root given as:
  //   sqrt(x) = x * 2.214 + x^2 * -1.098 + x^3 * 0.173
  auto sqrt = [](double x) -> double {
    return x*2.214 + x*x*(-1.098) + x*x*x*0.173;
  };
  VecInt2D F{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  std::vector<int> h(img.size(), 0), v(img.size(), 0), Ix, Iy;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // rot = image << (i*64+j)
      // create a copy of img as rotation using std::rotate always happens in-place
      auto rot = std::vector<int>(img);
      cyclicRotate(rot, i*img.size() + j);

      // h = rot * constant(scale, F[i][j])
      for (auto k = 0; k < rot.size(); ++k) h[k] = rot[k]*F[i][j];

      // v = rot * constant(scale, F[j][i])
      for (auto k = 0; k < rot.size(); ++k) v[k] = rot[k]*F[j][i];

      if (i==0 && j==0) {
        Ix = h;
        Iy = v;
      } else {
        // Ix = Ix + h
        std::transform(Ix.begin(), Ix.end(), h.begin(), Ix.begin(), std::plus<>());
        // Iy = Iv + v
        std::transform(Iy.begin(), Iy.end(), v.begin(), Iy.begin(), std::plus<>());
      }
    }
  }
  // Ix = Ix*Ix
  std::transform(Ix.begin(), Ix.end(), Ix.begin(), Ix.begin(), std::multiplies<>());
  // Iy = Iy*Iy
  std::transform(Iy.begin(), Iy.end(), Iy.begin(), Iy.begin(), std::multiplies<>());

  // result = Ix + Iy
  std::vector<int> result;
  result.reserve(Ix.size());
  std::transform(Ix.begin(), Ix.end(), Iy.begin(), std::back_inserter(result), std::plus<>());

  // result = sqrt(result[0]) + sqrt(result[1]) + ... + sqrt(result[N])
  std::transform(result.begin(), result.end(), result.begin(), [&sqrt](int elem) {
    return sqrt(elem);
  });

  return result;
}

