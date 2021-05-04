//
// Created by Moritz Winger on 03.05.21.
//

#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_PLAINTEXTNORM_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_PLAINTEXTNORM_H_

//
// Created by Moritz Winger on 03.05.21.
//
#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#include <seal/plaintext.h>

/// Calculate the infinity norm (max coeff) of a SEAL Plaintext
/// \param seal::Plaintext ptxt
int64_t plaintext_norm(seal::Plaintext ptxt);

#endif
#endif //AST_OPTIMIZER_SRC_UTILITIES_PLAINTEXTNORM_H_
