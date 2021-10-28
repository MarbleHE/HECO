#ifndef AST_OPTIMIZER_QUADRATICPOLYNOMIAL_H
#define AST_OPTIMIZER_QUADRATICPOLYNOMIAL_H

#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

std::vector<int64_t> encryptedQuadraticPolynomialPorcupine(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> c,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree
);

///  For a quadratic polynomial (from the porcupine paper)
/// This is the naive non-batched version
std::vector<int> encryptedQuadraticPolynomialNaive(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> c,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree
);

#endif//HAVE_SEAL_BFV

#endif//AST_OPTIMIZER_QUADRATICPOLYNOMIAL_H
