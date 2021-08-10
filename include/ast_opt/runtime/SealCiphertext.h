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
  std::unique_ptr<SealCiphertext> clone_impl() const;

 public:
  ~SealCiphertext() override = default;

  SealCiphertext(const SealCiphertext &other); // copy constructor

  SealCiphertext(SealCiphertext &&other) noexcept;  // copy assignment

  SealCiphertext &operator=(const SealCiphertext &other);  // move constructor

  SealCiphertext &operator=(SealCiphertext &&other);  // move assignment

  /// Creates a new SealCiphertext: a wrapper around the seal::Ciphertext class.
  /// \param sealFactory The factory that created this ciphertext.
  explicit SealCiphertext(const std::reference_wrapper<const SealCiphertextFactory> sealFactory);

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A const reference) to the underlying seal::Ciphertext.
  [[nodiscard]] const seal::Ciphertext &getCiphertext() const;

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A reference) to the underlying seal::Ciphertext.
  seal::Ciphertext &getCiphertext();

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> multiply(const AbstractCiphertext &operand) const override;

  void multiplyInplace(const AbstractCiphertext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> add(const AbstractCiphertext &operand) const override;

  void addInplace(const AbstractCiphertext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> subtract(const AbstractCiphertext &operand) const  override;

  void subtractInplace(const AbstractCiphertext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> rotateRows(int steps)const  override;

  void rotateRowsInplace(int steps) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> modSwitch(int num) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> multiplyPlain(const ICleartext &operand) const override;

  void multiplyPlainInplace(const ICleartext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> addPlain(const ICleartext &operand)const override;

  void addPlainInplace(const ICleartext &operand) override;

  [[nodiscard]] std::unique_ptr<AbstractCiphertext> subtractPlain(const ICleartext &operand)  const override;

  void subtractPlainInplace(const ICleartext &operand) override;

  std::unique_ptr<AbstractCiphertext> clone() const override;

  void add_inplace(const AbstractValue &other) override;

  void subtract_inplace(const AbstractValue &other) override;

  void multiply_inplace(const AbstractValue &other) override;

  void divide_inplace(const AbstractValue &other) override;

  void modulo_inplace(const AbstractValue &other) override;

  void logicalAnd_inplace(const AbstractValue &other) override;

  void logicalOr_inplace(const AbstractValue &other) override;

  void logicalLess_inplace(const AbstractValue &other) override;

  void logicalLessEqual_inplace(const AbstractValue &other) override;

  void logicalGreater_inplace(const AbstractValue &other) override;

  void logicalGreaterEqual_inplace(const AbstractValue &other) override;

  void logicalEqual_inplace(const AbstractValue &other) override;

  void logicalNotEqual_inplace(const AbstractValue &other) override;

  void logicalNot_inplace() override;

  void bitwiseAnd_inplace(const AbstractValue &other) override;

  void bitwiseXor_inplace(const AbstractValue &other) override;

  void bitwiseOr_inplace(const AbstractValue &other) override;

  void bitwiseNot_inplace() override;

  [[nodiscard]] const SealCiphertextFactory &getFactory() const override;
  int noiseBits() const;
};

#endif
#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SEALCIPHERTEXT_H_
