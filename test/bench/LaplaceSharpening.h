#ifndef AST_OPTIMIZER_BENCH_LAPLACESHARPENING_H_
#define AST_OPTIMIZER_BENCH_LAPLACESHARPENING_H_

#include <vector>
#include "MultiTimer.h"
#include <stdint.h>

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
std::vector<int> laplacianSharpening(const std::vector<int> &img);

/// Encrypted LaplacianSharpening, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedLaplacianSharpening(MultiTimer &timer,
                                                  const std::vector<int> &img,
                                                  size_t poly_modulus_degree,
                                                  bool encrypt_weights = false);

/// Encrypted LaplacianSharpening naively encrypting each pixel as an individual ctxt
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int> encryptedNaiveLaplaceSharpening(
    MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights = false);

/// Encrypted LaplacianSharpening naively encrypting img into one ctxt but using naive index access via rotation
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedNaiveBatchedLaplacianSharpening(
    MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights = false);

#endif //AST_OPTIMIZER_BENCH_LAPLACESHARPENING_H_
