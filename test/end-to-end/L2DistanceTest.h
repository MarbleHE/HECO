#ifndef AST_OPTIMIZER_L2DISTANCETEST_H
#define AST_OPTIMIZER_L2DISTANCETEST_H

#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

/// Output is squared to elide square root
/// For 4-element distance
/// Ciphertext l2_distance(Ciphertext c0, Ciphertext c1)
///     Ciphertext c2 = sub(c1, c0)/
///     Ciphertext c3 = square(c2)/
///     c3 = relinearize(c3)/
///     Ciphertext c4 = rotate(c3, 2)/
///     Ciphertext c5 = add(c3, c4)/
///     Ciphertext c6 = rotate(c4, 1)/
///     return add(c5, c6)/
int encryptedL2DistanceSquared_Porcupine(
        MultiTimer &timer, const std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree);

#endif //HAVE_SEAL_BFV

#endif//AST_OPTIMIZER_L2DISTANCETEST_H
