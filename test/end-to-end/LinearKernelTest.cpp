#include <vector>
#include <stdexcept>

/// Original, plain C++ program for linear kernel based classification
/// y = sgn(SUM_i^n w_i * y_i * k(x_i,x')
/// where k = L2 distance
///
/// Since sgn() is not supported in BFV, we omit it
/// The client can easily apply it after decryption
///
/// \param xs Existing Data Points (m vectors of size n)
/// \param ws Weights for data points (m scalars)
/// \param ys Labels for data points (m scalars)
/// \param x  Input data point (vector of size n)
/// \return linear kernel prediction
float linearKernel(const std::vector<std::vector<float>> &xs,
                     const std::vector<float> &ws,
                     const std::vector<float> &ys,
                     const std::vector<float> &x) {
  if (xs.empty() || ws.empty() || ys.empty() || xs[0].size()!=x.size() || ws.size()!=ys.size() || xs.size() != ys.size())
    throw std::runtime_error("Invalid input sizes for linear regression.");

  // for each input
  float sum = 0;
  for(size_t i = 0; i < xs.size(); ++i) {
    // compute L2 distance
    float k = 0;
    for (size_t j = 0; j < x.size(); ++j) {
      k += (x[j] - xs[i][j])*(x[j] - xs[i][j]);
    }
    // update sum
    sum += ws[i] * ys[i] * k;
  }

  // omitting sgn()
  return sum;
}
