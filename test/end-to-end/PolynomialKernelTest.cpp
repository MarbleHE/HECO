#include <cmath>
#include <vector>
#include <stdexcept>

/// Original, plain C++ program for polynomial kernel based classification
/// y = sgn(SUM_i^n w_i * y_i * k(x_i,x')
/// where k = (<x,y> +c)^d
///
/// Since sgn() is not supported in BFV, we omit it
/// The client can easily apply it after decryption
///
/// \param xs Existing Data Points (m vectors of size n)
/// \param ws Weights for data points (m scalars)
/// \param ys Labels for data points (m scalars)
/// \param c  Offset for polynomial kernel
/// \param d  Degree of polynomial kernel
/// \param x  Input data point (vector of size n)
/// \return linear kernel prediction
float polynomialKernel(const std::vector<std::vector<float>> &xs,
                       const std::vector<float> &ws,
                       const std::vector<float> &ys,
                       float c,
                       int d,
                       const std::vector<float> &x) {
  if (xs.empty() || ws.empty() || ys.empty() || xs[0].size()!=x.size() || ws.size()!=ys.size() || xs.size()!=ys.size())
    throw std::runtime_error("Invalid input sizes for linear regression.");

  // for each input
  float sum = 0;
  for (size_t i = 0; i < xs.size(); ++i) {
    // compute dot product
    float k = 0;
    for (size_t j = 0; j < x.size(); ++j) {
      sum += x[j]*xs[i][j];
    }
    // update sum
    sum += ws[i]*ys[i]*std::pow(k + c, d);
  }

  // omitting sgn()
  return sum;
}
