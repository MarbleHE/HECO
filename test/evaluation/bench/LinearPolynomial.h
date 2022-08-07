#ifndef AST_OPTIMIZER_LINEARPOLYNOMIAL_H
#define AST_OPTIMIZER_LINEARPOLYNOMIAL_H

#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

/// For a linear polynomial (from the porcupine paper)
/// Ciphertext linear_reg(Ciphertext a, Ciphertext b,
///                       Ciphertext x, Ciphertext y)
///     Ciphertext c1 = multiply(a, x)
///     c1 = relinearize(c1)
///     Ciphertext c2 = sub(y, c1)
///     return sub(c2, b)
std::vector<int64_t> encryptedLinearPolynomialPorcupine(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree
);

/// For a linear polynomial, HECO version (from the porcupine paper)
std::vector<int64_t> encryptedLinearPolynomialBatched(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree
);

/// For a linear polynomial (from the porcupine paper)
std::vector<int> encryptedLinearPolynomialNaive(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree
);

#endif

#endif//AST_OPTIMIZER_LINEARPOLYNOMIAL_H
