#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXT_H_

#include <memory>
#include "AbstractCiphertext.h"
#include "SealCiphertextFactory.h"

// forward declarations
class SealCiphertextFactory;

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

class SealCiphertext : public AbstractCiphertext {
 private:
  /// The encrypted data in this ciphertext wrapper.
  seal::Ciphertext ciphertext;
  std::unique_ptr<SealCiphertext> clone_impl();

 public:
  ~SealCiphertext() override = default;

  SealCiphertext(const SealCiphertext &other); // copy constructor

  SealCiphertext(SealCiphertext &&other) noexcept;  // copy assignment

  SealCiphertext &operator=(const SealCiphertext &other);  // move constructor

  SealCiphertext &operator=(SealCiphertext &&other) noexcept;  // move assignment

  /// Creates a new SealCiphertext: a wrapper around the seal::Ciphertext class.
  /// \param sealFactory The factory that created this ciphertext.
  explicit SealCiphertext(SealCiphertextFactory &sealFactory);

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A const reference) to the underlying seal::Ciphertext.
  [[nodiscard]] const seal::Ciphertext &getCiphertext() const;

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A reference) to the underlying seal::Ciphertext.
  seal::Ciphertext &getCiphertext();

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> multiply(AbstractCiphertext &operand) override;

  void multiplyInplace(AbstractCiphertext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> add(AbstractCiphertext &operand) override;

  void addInplace(AbstractCiphertext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> subtract(AbstractCiphertext &operand) override;

  void subtractInplace(AbstractCiphertext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> rotateRows(int steps) override;

  void rotateRowsInplace(int steps) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> multiplyPlain(ICleartext &operand) override;

  void multiplyPlainInplace(ICleartext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> addPlain(ICleartext &operand) override;

  void addPlainInplace(ICleartext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> subtractPlain(ICleartext &operand) override;

  void subtractPlainInplace(ICleartext &operand) override;

  std::unique_ptr<AbstractCiphertext> clone() override;

  void add(AbstractValue &other) override;

  void subtract(AbstractValue &other) override;

  void multiply(AbstractValue &other) override;

  void divide(AbstractValue &other) override;

  void modulo(AbstractValue &other) override;

  void logicalAnd(AbstractValue &other) override;

  void logicalOr(AbstractValue &other) override;

  void logicalLess(AbstractValue &other) override;

  void logicalLessEqual(AbstractValue &other) override;

  void logicalGreater(AbstractValue &other) override;

  void logicalGreaterEqual(AbstractValue &other) override;

  void logicalEqual(AbstractValue &other) override;

  void logicalNotEqual(AbstractValue &other) override;

  void logicalNot() override;

  void bitwiseAnd(AbstractValue &other) override;

  void bitwiseXor(AbstractValue &other) override;

  void bitwiseOr(AbstractValue &other) override;

  void bitwiseNot() override;

  SealCiphertextFactory &getFactory() override;

  [[nodiscard]] const SealCiphertextFactory &getFactory() const override;
};

#endif
#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXT_H_
