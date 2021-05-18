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
 // seal::Ciphertext _ciphertext;
  seal::Plaintext _plaintext;
  uint64_t _noise = 0; // current invariant noise scaled by coeff_modulus (i.e: we store actual_noise * coeff_modulus)
  uint64_t _noise_budget = 0; // current noise budget
  int ciphertext_size_ = 0; // ciphertext size: this gets bigger when multiplying and reset when relinearizing

  SimulatorCiphertext(std::reference_wrapper<const AbstractCiphertextFactory> simulatorFactory, seal::Plaintext ptxt);

  std::unique_ptr<SimulatorCiphertext> clone_impl() const;

 public:

  ~SimulatorCiphertext() override = default;

  SimulatorCiphertext(const SimulatorCiphertext &other); // copy constructor

  SimulatorCiphertext(SimulatorCiphertext &&other) noexcept;  // copy assignment

  SimulatorCiphertext &operator=(const SimulatorCiphertext &other);  // move constructor

  SimulatorCiphertext &operator=(SimulatorCiphertext &&other);  // move assignment

  /// Creates a new (empty) SimulatorCiphertext
  /// \param simulatorFactory The factory that created this ciphertext.
  explicit SimulatorCiphertext(const std::reference_wrapper<const AbstractCiphertextFactory> simulatorFactory);

  /// Creates a new SimulatorCiphertext from a SEAL Plaintext
  /// \param simulatorFactory The factory that created this ciphertext.
  /// \param ptxt The SEAL plaintext which we should use as the base for this ciphertext
  explicit SimulatorCiphertext(const SimulatorCiphertextFactory &simulatorFactory,
                               std::unique_ptr<seal::Plaintext> ptxt);


  // API function overrides from AbstractNoiseMeasuringCiphertext
  //TODO: documentation doxygen and push these to AbstractNoise measuring CTXT
  double getNoise() const;
  int64_t getNoiseBudget();
  int64_t getCoeffModulus();
  int noiseBits() const;

  // custom stuff
  const seal::Plaintext &getPlaintext() const;
  seal::Plaintext &getPlaintext();
  void relinearize();


  void createFresh(std::unique_ptr<seal::Plaintext> &plaintext);
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

  double getNoise(AbstractCiphertext &abstractCiphertext) const;

  std::unique_ptr<AbstractCiphertext> clone() const override;
  const SimulatorCiphertextFactory &getFactory() const override;

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A const reference) to the underlying seal::Ciphertext.
  [[nodiscard]] const seal::Ciphertext &getCiphertext() const;

  /// Gets the seal::Ciphertext associated with this SealCiphertext.
  /// \return (A reference) to the underlying seal::Ciphertext.
  seal::Ciphertext &getCiphertext();

  int64_t initialNoise() override;

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