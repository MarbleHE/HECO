#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXTFACTORY_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXTFACTORY_H_

#include "ast_opt/visitor/Runtime/CiphertextFactory.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

#define NUM_CTXT_SLOTS 16384

class SealCiphertextFactory : public CiphertextFactory {
 private:
  /// the SEAL context
  std::shared_ptr<seal::SEALContext> context;

  /// secret key, also used for (more efficient) encryption
  std::unique_ptr<seal::SecretKey> secretKey;

  /// public key
  std::unique_ptr<seal::PublicKey> publicKey;

  /// rotation keys
  std::unique_ptr<seal::GaloisKeys> galoisKeys;

  /// relinearization keys
  std::unique_ptr<seal::RelinKeys> relinKeys;

  std::unique_ptr<seal::BatchEncoder> encoder;

  std::unique_ptr<seal::Evaluator> evaluator;

  std::unique_ptr<seal::Encryptor> encryptor;

 public:
  std::unique_ptr<AbstractCiphertext> createCiphertext(std::vector<int64_t> &data) override;

  void setupContext();

  template<typename T>
  void expandVector(std::vector<T> &values);
};

#endif

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXTFACTORY_H_
