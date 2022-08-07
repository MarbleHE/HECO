#ifndef AST_OPTIMIZER_GYKERNEL_H
#define AST_OPTIMIZER_GYKERNEL_H
#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

/// Encrypted GyKernel, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedBatchedGyKernel(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights = false);

/// Encrypted GyKernel, ported from Porcupine, according to:
/// Ciphertext gy(Ciphertext c0, int h, int w)
///      Ciphertext c1 = rotate(c0, 1)
///      Ciphertext c2 = add(c0, c1)
///      Ciphertext c3 = rotate(c2, 1)
///      Ciphertext c4 = add(c2, c3)
///      Ciphertext c5 = rotate(c4, -1 + w)
///      Ciphertext c6 = rotate(c4, -1 -w)
///      return sub(c5, c6)
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \return transformed image
std::vector<int64_t> encryptedGyKernelPorcupine(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree);

/// Encrypted GyKernel, using vectors of ciphertexts.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \return transformed image
std::vector<int> encryptedNaiveGyKernel(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree);

#endif
#endif//AST_OPTIMIZER_GYKERNEL_H
