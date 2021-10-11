#ifndef AST_OPTIMIZER_ROBERTSCROSS_H
#define AST_OPTIMIZER_ROBERTSCROSS_H

#include "seal/seal.h"
#include "MultiTimer.h"

std::vector<int64_t> encryptedRobertsCrossPorcupine(
        MultiTimer &timer, std::vector<int> &img, size_t poly_modulus_degree);

#endif//AST_OPTIMIZER_ROBERTSCROSS_H
