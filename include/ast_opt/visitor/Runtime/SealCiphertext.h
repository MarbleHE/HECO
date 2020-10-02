#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXT_H_

#include <memory>
#include "AbstractCiphertext.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

class SealCiphertext : public AbstractCiphertext {
// private: // TODO
 public:
  /// the encrypted data in this ciphertext wrapper
  seal::Ciphertext ciphertext;

  std::shared_ptr<seal::SEALContext> context;

 public:

};

#endif
#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXT_H_
