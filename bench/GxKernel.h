#ifndef AST_OPTIMIZER_GXKERNEL_H
#define AST_OPTIMIZER_GXKERNEL_H

#include "seal/seal.h"
#include "MultiTimer.h"

/// Encrypted GxKernel, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedBatchedGxKernel(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights = false);

/// Encrypted GxKernel, ported from Porcupine, according to:
/// Ciphertext gx(Ciphertext c0, int h, int w)
///     Ciphertext c1 = rotate(c0, w)
///     Ciphertext c2 = add(c0, c1)
///     Ciphertext c3 = rotate(c2, -w)
///     Ciphertext c4 = add(c2, c3)
///     Ciphertext c5 = rotate(c4, 1)
///     Ciphertext c6 = rotate(c4, -1)
///     return sub(c5, c6)
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \return transformed image
std::vector<int64_t> encryptedBatchedGxKernelPorcupine(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree);

#endif//AST_OPTIMIZER_GXKERNEL_H
