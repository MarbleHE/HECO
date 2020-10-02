#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXTFACTORY_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXTFACTORY_H_

#include "ast_opt/visitor/Runtime/AbstractCiphertextFactory.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

class SealCiphertextFactory : public AbstractCiphertextFactory {
 private:
  /// The number of slots (i.e., maximum no. of elements) in a ciphertext.
  const uint ciphertextSlotSize = 16'384;

  /// The SEAL context.
  std::shared_ptr<seal::SEALContext> context;

  /// The secret key, also used for (more efficient) encryption.
  std::unique_ptr<seal::SecretKey> secretKey;

  /// The public key.
  std::unique_ptr<seal::PublicKey> publicKey;

  /// The rotation keys.
  std::unique_ptr<seal::GaloisKeys> galoisKeys;

  /// The relinearization keys.
  std::unique_ptr<seal::RelinKeys> relinKeys;

  /// The encoder helper object.
  std::unique_ptr<seal::BatchEncoder> encoder;

  /// The evaluator helper object.
  std::unique_ptr<seal::Evaluator> evaluator;

  /// The encryptor helper object.
  std::unique_ptr<seal::Encryptor> encryptor;

  /// The decryptor helper object.
  std::unique_ptr<seal::Decryptor> decryptor;

  /// Sets up the FHE scheme by creating a new context, setting required scheme parameters, generating keys, and
  /// instantiating the helper objects.
  void setupSealContext();

  /// Fills a given vector with its last element until it has ciphertextSlotSize elements.
  /// \tparam T The type of the elements that should be encoded in the ciphertext.
  /// \param values The vector that should be expanded by its last element.
  /// \throws std::runtime_error if the number of elements in values is larger than the size defined in
  /// ciphertextSlotSize.
  template<typename T>
  void expandVector(std::vector<T> &values);

 public:
  SealCiphertextFactory() = default;

  /// Creates a new SealCiphertextFactory whereat each ciphertext created by the factory can hold
  /// numElementsPerCiphertextSlot elements.
  /// \param numElementsPerCiphertextSlot The number of ciphertext slots of created ciphertexts. Must be a power of
  /// two, e.g., 4'096 or 8'192.
  explicit SealCiphertextFactory(uint numElementsPerCiphertextSlot);

  /// Gets the relinearization keys, required for relinearizing a ciphertext after a multiplication, that were
  /// precomputed in setupSealContext().
  /// \return (A const reference to) the seal::RelinKeys object.
  [[nodiscard]] const seal::RelinKeys &getRelinKeys() const;

  /// Gets the Galois keys, required for rotating a ciphertext, that were precomputed in setupSealContext().
  /// \return (A const reference to) the seal::GaloisKeys object.
  [[nodiscard]] const seal::GaloisKeys &getGaloisKeys() const;

  /// Gets the seal::Evaluator instance.
  /// \return (A reference to) the seal::Evaluator instance.
  [[nodiscard]] seal::Evaluator &getEvaluator() const;

  std::unique_ptr<AbstractCiphertext> createCiphertext(std::vector<int64_t> &data) override;

  void decryptCiphertext(AbstractCiphertext &abstractCiphertext, std::vector<int64_t> &ciphertextData) override;
};

#endif

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXTFACTORY_H_
