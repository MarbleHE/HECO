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


#endif
#endif//AST_OPTIMIZER_DOTPRODUCT_H
