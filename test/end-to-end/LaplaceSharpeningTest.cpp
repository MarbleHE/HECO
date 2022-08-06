#include <random>
#include <set>
#include "gtest/gtest.h"

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838
// use a 4x4 matrix for the tests
#define MATRIX_SIZE 4

/// Original, plain C++ program for LaplacianSharpening
/// This uses a 3x3 Kernel and applies it by sliding across the 2D image
///             | 1  1  1 |
///   w = -1 *  | 1 -8  1 |
///             | 1  1  1 |
/// No padding is used, so the image is 2px smaller in each dimension
/// The filtered image is then added to the original image at 50% intensity.
///
/// This program is inspired by RAMPARTS, presented by Archer et al. in 2019
/// The only modification that we do is instead of computing
///     img2[x][y] = img[x][y] - (value/2),
/// we compute
///     img2[x][y] = 2*img[x][y] - value
/// to avoid division which is unsupported in BFV
/// The client can easily divide the result by two after decryption
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> laplacianSharpening(const std::vector<int> &img)
{
    const auto imgSize = (int)std::ceil(std::sqrt(img.size()));
    std::vector<std::vector<int>> weightMatrix = { { 1, 1, 1 }, { 1, -8, 1 }, { 1, 1, 1 } };
    std::vector<int> img2(img.begin(), img.end());
    for (int x = 1; x < imgSize - 1; ++x)
    {
        for (int y = 1; y < imgSize - 1; ++y)
        {
            int value = 0;
            for (int j = -1; j < 2; ++j)
            {
                for (int i = -1; i < 2; ++i)
                {
                    value = value + weightMatrix.at(i + 1).at(j + 1) * img.at((x + i) * imgSize + (y + j));
                }
            }
            img2[imgSize * x + y] = 2 * img.at(x * imgSize + y) - value;
        }
    }
    return img2;
}