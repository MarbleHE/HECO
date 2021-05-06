#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SIMULATORCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SIMULATORCIPHERTEXT_H_

#include <memory>
#include "AbstractCiphertext.h"
#include "SimulatorCiphertextFactory.h"
#include "AbstractNoiseMeasuringCiphertext.h"

// forward declarations
class SimulatorCiphertextFactory;

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>


class SimulatorCiphertext : public AbstractNoiseMeasuringCiphertext {
 private:
  SimulatorCiphertext(SimulatorCiphertextFactory &simulatorFactory, seal::Plaintext ptxt);
  seal::Ciphertext _ciphertext;
  seal::Plaintext _plaintext;
  double _noise = 0; // current invariant noise
  double _noise_budget = 0; // current noise budget
  std::unique_ptr<SimulatorCiphertext> clone_impl();
  int ciphertext_size_ = 0; // ciphertext size: this gets bigger when multiplying and reset when relinearizing

 public:

  ~SimulatorCiphertext() override = default;

  SimulatorCiphertext(const SimulatorCiphertext &other); // copy constructor

  SimulatorCiphertext(SimulatorCiphertext &&other) noexcept;  // copy assignment

  SimulatorCiphertext &operator=(const SimulatorCiphertext &other);  // move constructor

  SimulatorCiphertext &operator=(SimulatorCiphertext &&other) noexcept;  // move assignment

  /// Creates a new SimulatorCiphertext: a wrapper around the seal::Ciphertext class.
  /// \param simulatorFactory The factory that created this ciphertext.
  explicit SimulatorCiphertext(SimulatorCiphertextFactory &simulatorFactory);

  //moritz
  explicit SimulatorCiphertext(SimulatorCiphertextFactory &simulatorFactory, std::unique_ptr<seal::Plaintext> ptxt);


  void createFresh(std::unique_ptr<seal::Plaintext> &plaintext);
  std::unique_ptr<AbstractCiphertext> multiply(AbstractCiphertext &operand) override;
  void multiplyInplace(AbstractCiphertext &operand) override;
  std::unique_ptr<AbstractCiphertext> multiplyPlain(ICleartext &operand) override;
  void multiplyPlainInplace(ICleartext &operand) override;
  std::unique_ptr<AbstractCiphertext> add(AbstractCiphertext &operand) override;
  void addInplace(AbstractCiphertext &operand) override;
  std::unique_ptr<AbstractCiphertext> addPlain(ICleartext &operand) override;
  void addPlainInplace(ICleartext &operand) override;
  std::unique_ptr<AbstractCiphertext> subtract(AbstractCiphertext &operand) override;
  void subtractInplace(AbstractCiphertext &operand) override;
  std::unique_ptr<AbstractCiphertext> subtractPlain(ICleartext &operand) override;
  void subtractPlainInplace(ICleartext &operand) override;
  std::unique_ptr<AbstractCiphertext> rotateRows(int steps) override;
  void rotateRowsInplace(int steps) override;
  void noiseBits() override;

  std::unique_ptr<AbstractCiphertext> clone() override;
  SimulatorCiphertextFactory &getFactory() override;
  const SimulatorCiphertextFactory &getFactory() const override;

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A const reference) to the underlying seal::Ciphertext.
  [[nodiscard]] const seal::Ciphertext &getCiphertext() const;

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A reference) to the underlying seal::Ciphertext.
  seal::Ciphertext &getCiphertext();

  void relinearize();

  int64_t initialNoise() override;

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
  const seal::Plaintext &getPlaintext() const;
  seal::Plaintext &getPlaintext();
  double &getNoise();
};

#endif
#endif