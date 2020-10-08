#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALPLAINTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALPLAINTEXT_H_

#include "ast_opt/visitor/runtime/AbstractPlaintext.h"
#include "SealCiphertextFactory.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

class SealPlaintext : public AbstractPlaintext {
 private:
  /// The encoded data in this plaintext wrapper.
  seal::Plaintext plaintext;

  /// A reference to the factory that created this ciphertext.
  SealCiphertextFactory &factory;

 public:
  ~SealPlaintext() override = default;

  explicit SealPlaintext(SealCiphertextFactory &sealFactory);

  [[nodiscard]] const seal::Plaintext &getPlaintext() const;

  seal::Plaintext &getPlaintext();
};

#endif

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALPLAINTEXT_H_
