#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXTFACTORY_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXTFACTORY_H_

#include <initializer_list>
#include <string>

class AbstractCiphertext;

class AbstractCiphertextFactory {
 public:
  /// Creates a new AbstractCiphertext with the instantiated concrete ciphertext factory. The last value is used to fill
  /// up the remaining slots of the ciphertext, in case that the given data does not use all slots.
  /// \param data The data that should be encrypted in the ciphertext.
  /// \return (A std::unique_ptr) to the created AbstractCiphertext.
  virtual std::unique_ptr<AbstractCiphertext> createCiphertext(std::vector<int64_t> &data) = 0;

  /// Creates a new AbstractCiphertext with the instantiated concrete ciphertext factory. The given value is written
  /// into each slot of the ciphertext.
  /// \param data The value that should be encrypted in the ciphertext.
  /// \return (A std::unique_ptr) to the created AbstractCiphertext.
  virtual std::unique_ptr<AbstractCiphertext> createCiphertext(int64_t data) = 0;

  /// Decrypts a given Abstractciphertext and writes the decrypted and decoded results into the given vector reference.
  /// \param abstractCiphertext (A reference to) the ciphertext that should be decrypted and decoded.
  /// \param ciphertextData (A reference to) the vector where the decrypted and decoded values are written to.
  virtual void decryptCiphertext(AbstractCiphertext &abstractCiphertext, std::vector<int64_t> &ciphertextData) = 0;

  /// Generates and returns the textual representation of the given ciphertext.
  /// \param abstractCiphertext The ciphertext for that a textual representation should be generated for.
  /// \return (A std::string with) the textual representation of the encrypted ciphertext elements.
  virtual std::string getString(AbstractCiphertext &abstractCiphertext) = 0;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXTFACTORY_H_
