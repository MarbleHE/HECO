#ifndef AST_OPTIMIZER_HAMMINGDISTANCE_H
#define AST_OPTIMIZER_HAMMINGDISTANCE_H

#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

uint64_t encryptedNaiveHammingDistance(MultiTimer &timer, const std::vector<bool> &a, const std::vector<bool> &b, size_t poly_modulus_degree);

/// Computes the encrypted hamming distance between two vectors of booleans
/// Note: Hamming distance over binary vectors can be computed semi-efficiently in Z_p by using NEQ = XOR = (a-b)^2
/// \param a vector of size n
/// \param b vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_both By default, both vectors are encrypted. If set to false, b is plaintext
/// \return
uint64_t encryptedBatchedHammingDistance(
        MultiTimer &timer,
        const std::vector<bool> &a,
        const std::vector<bool> &b,
        size_t poly_modulus_degree,
        bool encrypt_both = true
        );

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
int encryptedHammingDistancePorcupine(
        MultiTimer &timer, const std::vector<bool> &a, const std::vector<bool> &b, size_t poly_modulus_degree);

#endif
#endif//AST_OPTIMIZER_HAMMINGDISTANCE_H
