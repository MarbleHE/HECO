#ifndef AST_OPTIMIZER_DOTPRODUCT_H
#define AST_OPTIMIZER_DOTPRODUCT_H
#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

/// Batched version of dot product, where one ciphertext contains one vector.
/// \param x a vector of size n
/// \param y a vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials. Has to larger than n.
int encryptedDotProductBatched(MultiTimer &timer, std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree);

/// Naive version of dot product, where one ciphertext contains one value.
/// \param x a vector of size n
/// \param y a vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials. Has to larger than n.
int encryptedDotProductNaive(MultiTimer &timer, std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree);

/// Porcupine version of dot product, which is based on the porcupine pseudocode. Only accepts vectors of length 8:
/// Ciphertext dot_product(Ciphertext c0, Plaintext p0)
///     Ciphertext c1 = multiply(c0, p0)
///     Ciphertext c2 = rotate(c1, 2)
///     Ciphertext c3 = add(c1, c2)
///     Ciphertext c4 = rotate(c3, 4)
///     Ciphertext c5 = add(c3, c4)
///     Ciphertext c6 = rotate(c5, 1)
///     return add(c5, c6)
/// \param x a vector of size n
/// \param y a vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials. Has to larger than n.
int encryptedDotProductPorcupine(MultiTimer &timer, std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree);


#endif
#endif//AST_OPTIMIZER_DOTPRODUCT_H
