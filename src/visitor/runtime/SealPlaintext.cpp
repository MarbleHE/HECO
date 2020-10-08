#include "ast_opt/visitor/runtime/SealPlaintext.h"

#ifdef HAVE_SEAL_BFV

const seal::Plaintext &SealPlaintext::getPlaintext() const {
  return plaintext;
}

SealPlaintext::SealPlaintext(SealCiphertextFactory &sealFactory) : factory(sealFactory) {}

seal::Plaintext &SealPlaintext::getPlaintext() {
  return plaintext;
}

#endif
