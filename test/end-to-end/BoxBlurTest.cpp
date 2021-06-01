#include <cmath>
#include "gtest/gtest.h"

/// Original, plain C++ program for a naive Box blur
/// This uses a 3x3 Kernel and applies it by sliding across the 2D image
///             | 1  1  1 |
///   w = 1/9 * | 1  1  1 |
///             | 1  1  1 |
/// No padding is used, so the image is 2px smaller in each dimension
/// To avoid division (which is unsupported in BFV) we omit the 1/9
/// The client can easily divide the result by nine after decryption
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> naiveBoxBlur(std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  std::vector<std::vector<int>> weightMatrix = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
  std::vector<int> img2(img.begin(), img.end());
  for (int x = 1; x < imgSize - 1; ++x) {
    for (int y = 1; y < imgSize - 1; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
          value = value + weightMatrix.at(i + 1).at(j + 1)*img.at((x + i)*imgSize + (y + j));
        }
      }
      img2[imgSize*x + y] = value;
    }
  }
  return img2;
}

/// Original, plain C++ program for a fast Box blur
/// Instead of using a using a 2D Kernel, this uses two 1D Kernels
///         | 1 |                         | 1  1  1 |
///   1/3 * | 1 | * 1/3 [1  1  1] = 1/9 * | 1  1  1 |
///         | 1 |                         | 1  1  1 |
/// No padding is used, so the image is 2px smaller in each dimension
///
/// This version is based http://amritamaz.net/blog/understanding-box-blur
/// The separation of the kernels allows it to do horizontal and vertical blurs separately.
/// It also uses the fact that the kernel weights are the same in each position
/// and simply adds and removes pixels from a running value computation.
///
/// To avoid division (which is unsupported in BFV) we omit the two 1/3
/// The client can easily divide the result by nine after decryption
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> fastBoxBlur(std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  std::vector<int> img2(img.begin(), img.end());

  // Horizontal Kernel: for each row y
  for (int y = 1; y < imgSize - 1; ++y) {
    // Get kernel for second (because of padding) pixel of row y
    int value = img.at(0*imgSize + y) + img.at(1*imgSize + y) + img.at(2*imgSize + y);
    // Division that would usually happen here is omitted
    img2[1*imgSize + y] = value;

    // Go through the rest of row y
    for (int x = 2; x < imgSize - 1; ++x) {
      // remove the previous pixel
      value -= img.at((x-2)*imgSize + y); //x = middle of current kernel, x-2 = one to the left of kernel
      // add the new pixel
      value += img.at((x+1)*imgSize + y); //x = right pixel of previous kernel, x+1 = right pixel of new kernel
      // save result
      img2[x*imgSize + y] = value;
    }

  }

  // Now apply the vertical kernel to img2

  // Create new output image
  std::vector<int> img3(img2.begin(), img2.end());

  // Vertical Kernel: for each column x
  for (int x = 1; x < imgSize - 1; ++x) {
    // Get kernel for second (because of padding) pixel of column x
    int value = img2.at(x*imgSize + 0) + img2.at(x*imgSize + 1) + img2.at(x*imgSize + 2);
    // Division that would usually happen here is omitted
    img3[x*imgSize + 1] = value;

    // Go through the rest of column x
    for (int y = 2; y < imgSize - 1; ++y) {
      // remove the previous pixel
      value -= img2.at(x*imgSize + y-2); //y = middle of current kernel, y-2 = one to the left of kernel
      // add the new pixel
      value += img2.at(x*imgSize + y+1); //y = right pixel of previous kernel, y+1 = right pixel of new kernel
      // save result
      img3[x*imgSize + y] *= value;
    }

  }
  return img3;
}


//TODO: Test to assert that naivBoxBlur and fastBoxBlur actually compute the same thing!
