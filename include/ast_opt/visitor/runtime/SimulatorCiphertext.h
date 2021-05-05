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
#include "ast_opt/utilities/seal_2.3.0/biguint.h"
#include "ast_opt/utilities/seal_2.3.0/memorypoolhandle.h"
#include "ast_opt/utilities/seal_2.3.0/bigpoly.h"


class SimulatorCiphertext : public AbstractNoiseMeasuringCiphertext {
 private:
  seal::Ciphertext ciphertext;
  double _noise = 0; // current invariant noise
  double noise_budget = 0; // current noise budget
  std::unique_ptr<SimulatorCiphertext> clone_impl();
  int64_t coeff_modulus_ = 0;
  int coeff_modulus_bit_count_ = 0;
  int ciphertext_size_ = 0;
  int number_of_mults = 0;

 public:

  ~SimulatorCiphertext() override = default;

  SimulatorCiphertext(const SimulatorCiphertext &other); // copy constructor

  SimulatorCiphertext(SimulatorCiphertext &&other) noexcept;  // copy assignment

  SimulatorCiphertext &operator=(const SimulatorCiphertext &other);  // move constructor

  SimulatorCiphertext &operator=(SimulatorCiphertext &&other) noexcept;  // move assignment

  /// Creates a new SimulatorCiphertext: a wrapper around the seal::Ciphertext class.
  /// \param simulatorFactory The factory that created this ciphertext.
  explicit SimulatorCiphertext(SimulatorCiphertextFactory &simulatorFactory);


  void createFresh(ICleartext &operand);
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
};

#endif
#endif