#ifndef AST_OPTIMIZER_ROBERTSCROSS_H
#define AST_OPTIMIZER_ROBERTSCROSS_H

#ifdef HAVE_SEAL_BFV
#include "seal/seal.h"
#include "MultiTimer.h"

std::vector<int64_t> encryptedRobertsCrossPorcupine(
        MultiTimer &timer, std::vector<int> &img, size_t poly_modulus_degree);

#endif
#endif//AST_OPTIMIZER_ROBERTSCROSS_H
