#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "heco/legacy_ast/ast_parser/Parser.h"
#include "heco/legacy_ast/ast_utilities/Scope.h"

/// Original, plain C++ program for a naive Gy Kernel
/// This uses a 3x3 Kernel and applies it by sliding across the 2D image
///        | +1  +2 +1 |
///   w =  |  0  0   0 |
///        | -1 -2  -1 |
/// This uses wrap-around padding
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> naiveGyKernel(const std::vector<int> &img)
{
    const auto imgSize = (int)std::ceil(std::sqrt(img.size()));
    // Encoded same as images
    std::vector<std::vector<int>> weightMatrix = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
    std::vector<int> img2(img.begin(), img.end());
    for (int x = 0; x < imgSize; ++x)
    {
        for (int y = 0; y < imgSize; ++y)
        {
            int value = 0;
            for (int j = -1; j < 2; ++j)
            {
                for (int i = -1; i < 2; ++i)
                {
                    value =
                        value + weightMatrix.at(i + 1).at(j + 1) * img.at(((x + i) * imgSize + (y + j)) % img.size());
                }
            }
            img2[imgSize * x + y] = value;
        }
    }
    return img2;
}

/// Original, plain C++ program for a faster Gy Kernel
/// Instead of using a using a 2D Kernel, this uses two 1D Kernels
///    | +1 |                  | +1 +2  +1 |
///    |  0 | * [+1 +2 +1] =   |  0  0   0 |
///    | -1 |                  | -1 -2  -1 |
/// This uses wrap-around padding
///
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<int> fastGyKernel(const std::vector<int> &img)
{
    const auto imgSize = (int)std::ceil(std::sqrt(img.size()));
    std::vector<int> img2(img.begin(), img.end());

    // First apply [+1  2  +1]
    for (int y = 0; y < imgSize; ++y)
    {
        // Get kernel for first pixel of row y, using padding
        int value = img.at((-1 * imgSize + y) % img.size()) + 2 * img.at((0 * imgSize + y) % img.size()) +
                    img.at(1 * imgSize + y);
        img2[0 * imgSize + y] = value;

        // Go through the rest of row y
        for (int x = 1; x < imgSize; ++x)
        {
            // remove the leftmost pixel (old weight +1, now outside kernel)
            // x = middle of current kernel, x-2 = one to the left of kernel
            value -= img.at(((x - 2) * imgSize + y) % img.size());

            // subtract the left pixel (old weight +2, new weight +1)
            // x = middle kernel, x-1 = left element of kernel
            value -= img.at(((x - 1) * imgSize + y) % img.size());

            // add the middle pixel to double it (old weight +1, new weight +2)
            // x = right pixel of previous kernel = middle pixel of new kernel
            value += img.at(((x)*imgSize + y) % img.size());

            // finally, add the right most pixel (no old weight, new weight +1)
            // x = right pixel of previous kernel, x+1 = right pixel of new kernel
            value += img.at(((x + 1) * imgSize + y) % img.size());

            // save result
            img2[x * imgSize + y] = value;
        }
    }

    // Now apply the vertical kernel to img2
    // | +1 |
    // |  0 |
    // | -1 |

    // Create new output image
    std::vector<int> img3(img2.begin(), img2.end());

    // Vertical Kernel: for each column x
    for (int x = 0; x < imgSize; ++x)
    {
        // Get kernel for first pixel of column x with padding
        int value = img2.at((x * imgSize - 1) % img.size()) - img2.at(x * imgSize + 1);
        // Division that would usually happen here is omitted
        img3[x * imgSize + 0] = value;

        // Go through the rest of column x
        for (int y = 1; y < imgSize; ++y)
        {
            // remove the leftmost pixel (old weight +1, now outside kernel)
            // y = middle of current kernel, y-2 = one to the left of kernel
            value -= img2.at((x * imgSize + y - 2) % img.size());

            // add the left pixel (old weight 0, new weight +1)
            // x = middle kernel, x-1 = left element of kernel
            value += img2.at((x * imgSize + y - 1) % img.size());

            // add one copy of the middle pixel to cancel out (old weight -1, new weight 0)
            // y = right pixel of previous kernel = middle pixel of new kernel
            value += img2.at((x * imgSize + y) % img.size());

            // finally, subtract the right most pixel (no old weight, new weight +1)
            // y = right pixel of previous kernel, y+1 = right pixel of new kernel
            value -= img2.at((x * imgSize + y + 1) % img.size());

            // save result
            img3[x * imgSize + y] = value;
        }
    }
    return img3;
}

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838
// use a 4x4 matrix for the tests
#define MATRIX_SIZE 4

class GyKernelTest : public ::testing::Test
{ /* NOLINT (predictable sequence expected) */
protected:
    std::default_random_engine randomEngine;
    std::uniform_int_distribution<int> myUnifIntDist;

    void SetUp() override
    {
        randomEngine = std::default_random_engine(RAND_SEED); /* NOLINT (predictable sequence expected) */
        // the supported number range must be according to the FHE scheme parameters to not wrap around the modulo
        myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);
    }

public:
    void resetRandomEngine()
    {
        randomEngine.seed(RAND_SEED);
    }

    void getInputMatrix(size_t size, std::vector<std::vector<int>> &destination)
    {
        // reset the RNG to make sure that every call to this method results in the same numbers
        resetRandomEngine();
        // make sure we clear desination vector before, otherwise resize could end up appending elements
        destination.clear();
        destination.resize(size, std::vector<int>(size));
        for (size_t i = 0; i < size; ++i)
        {
            for (size_t j = 0; j < size; ++j)
            {
                destination[i][j] = myUnifIntDist(randomEngine);
            }
        }
    }

    void getInputMatrix(size_t size, std::vector<int> &destination)
    {
        // make sure we clear desination vector before, otherwise resize could end up appending elements
        destination.clear();
        std::vector<std::vector<int>> data;
        getInputMatrix(size, data);
        std::size_t total_size = 0;
        for (const auto &sub : data)
            total_size += sub.size();
        destination.reserve(total_size);
        for (const auto &sub : data)
            destination.insert(destination.end(), sub.begin(), sub.end());
    }

    void printMatrix(size_t size, std::vector<int> &matrix)
    {
        for (int64_t row = (int64_t)size - 1; row >= 0; --row)
        {
            std::cout << matrix.at(0 * size + row);
            for (size_t col = 1; col < size; ++col)
            {
                std::cout << "\t" << matrix.at(col * size + row);
            }
            std::cout << std::endl;
        }
    }
};

/// Test to ensure that naiveGyKernel and fastGyKernel actually compute the same thing!
TEST_F(GyKernelTest, NaiveGyKernel_FastGyKernel_Equivalence)
{ /* NOLINT */

    size_t size = 4;
    std::vector<int> img;
    GyKernelTest::getInputMatrix(size, img);
    // std::cout << "img:" << std::endl;
    // printMatrix(size, img);

    auto naive = naiveGyKernel(img);
    // std::cout << "naive:" << std::endl;
    // printMatrix(size, naive);

    auto fast = fastGyKernel(img);
    // std::cout << "fast:" << std::endl;
    // printMatrix(size, fast);

    EXPECT_EQ(fast, naive);
}