#ifndef AST_OPTIMIZER_ROBERTSCROSS_H
#define AST_OPTIMIZER_ROBERTSCROSS_H

#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

/// Encrypted RobertsCross, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedBatchedRobertsCross(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights = false);

std::vector<int64_t> encryptedRobertsCrossPorcupine(
        MultiTimer &timer, std::vector<int> &img, size_t poly_modulus_degree);

#endif
#endif//AST_OPTIMIZER_ROBERTSCROSS_H
