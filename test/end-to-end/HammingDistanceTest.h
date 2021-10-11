#ifndef AST_OPTIMIZER_HAMMINGDISTANCETEST_H
#define AST_OPTIMIZER_HAMMINGDISTANCETEST_H

#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

/// For 4-element hamming distance
/// Ciphertext hamming_distance(Ciphertext c0, Ciphertext c1)
///     Plaintext p0(N, 2) // N is the number of slots
///     Ciphertext c2 = add(c1, c0)
///     Ciphertext c2_ = negate(c2)
///     Ciphertext c3 = add(c2_, p0)
///     Ciphertext c4 = multiply(c3, c2)
///     c4 = relinearize(c4)
///     Ciphertext c5 = rotate(c4, 2)
///     Ciphertext c6 = add(c4, c5)
///     Ciphertext c7 = rotate(c6, 1)
///     return add(c6, c7)
int encryptedHammingDistance_Porcupine(
        MultiTimer &timer, const std::vector<bool> &a, const std::vector<bool> &b, size_t poly_modulus_degree);

#endif //HAVE_SEAL_BFV

#endif//AST_OPTIMIZER_HAMMINGDISTANCETEST_H
