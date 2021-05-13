#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_DUMMYCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_DUMMYCIPHERTEXT_H_

#include <memory>
#include "AbstractCiphertext.h"
#include "DummyCiphertextFactory.h"

// forward declarations
class DummyCiphertextFactory;

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

class DummyCiphertext : public AbstractCiphertext {
 private:
  std::vector<int64_t> _data;

  std::unique_ptr<DummyCiphertext> clone_impl() const;

 public:
  ~DummyCiphertext() override = default;

  DummyCiphertext(const DummyCiphertext &other); // copy constructor

  DummyCiphertext(DummyCiphertext &&other) noexcept;  // copy assignment

  DummyCiphertext &operator=(const DummyCiphertext &other);  // move constructor

  DummyCiphertext &operator=(DummyCiphertext &&other);  // move assignment

  /// Creates a new (empty) DummyCiphertext
  /// \param simulatorFactory The factory that created this ciphertext.
  explicit DummyCiphertext(const std::reference_wrapper<const AbstractCiphertextFactory> dummyFactory);

  // return data vector
  std::vector<int64_t> getData();

  void createFresh(const std::vector<int64_t> &data);
  std::unique_ptr<AbstractCiphertext> multiply(const AbstractCiphertext &operand) const override;
  void multiplyInplace(const AbstractCiphertext &operand) override;
  std::unique_ptr<AbstractCiphertext> multiplyPlain(const ICleartext &operand) const override;
  void multiplyPlainInplace(const ICleartext &operand) override;
  std::unique_ptr<AbstractCiphertext> add(const AbstractCiphertext &operand) const override;
  void addInplace(const AbstractCiphertext &operand) override;
  std::unique_ptr<AbstractCiphertext> addPlain(const ICleartext &operand) const override;
  void addPlainInplace(const ICleartext &operand) override;
  std::unique_ptr<AbstractCiphertext> subtract(const AbstractCiphertext &operand) const override;
  void subtractInplace(const AbstractCiphertext &operand) override;
  std::unique_ptr<AbstractCiphertext> subtractPlain(const ICleartext &operand) const override;
  void subtractPlainInplace(const ICleartext &operand) override;
  std::unique_ptr<AbstractCiphertext> rotateRows(int steps) const override;
  void rotateRowsInplace(int steps) override;

  std::unique_ptr<AbstractCiphertext> clone() const override;
  const DummyCiphertextFactory &getFactory() const override;

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A const reference) to the underlying seal::Ciphertext.
  [[nodiscard]] const seal::Ciphertext &getCiphertext() const;

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A reference) to the underlying seal::Ciphertext.
  //seal::Ciphertext &getCiphertext();

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
};



#endif
#endif