#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SIMULATORCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SIMULATORCIPHERTEXT_H_

#include <memory>
#include "AbstractCiphertext.h"
#include "SimulatorCiphertextFactory.h"

// forward declarations
class SimulatorCiphertextFactory;

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#include <ast_opt/utilities/seal2.3.0_util/biguint.h>
#include "AbstractNoiseMeasuringCiphertext.h"

class SimulatorCiphertext : public AbstractNoiseMeasuringCiphertext {
 private:
  seal::Ciphertext ciphertext;
  seal_util::BigUInt _noise; // current invariant noise
  double noise_budget = 0; // current noise budget
  std::unique_ptr<SimulatorCiphertext> clone_impl();

  // added by MW
  seal_util::BigUInt noise_;

  seal_util::BigUInt coeff_modulus_;

  int coeff_modulus_bit_count_ = 0;

  int ciphertext_size_ = 0;

  seal::EncryptionParameters parms_;


 public:
  ~SimulatorCiphertext() override = default;


/**
        Creates a simulation of a ciphertext encrypted with the specified encryption
        parameters and given invariant noise budget. The given noise budget must be
        at least zero, and at most the significant bit count of the coefficient
        modulus minus two.

        @param[in] acf
        @param[in] parms The encryption parameters
        @param[in] noise_budget The invariant noise budget of the created ciphertext
        @param[in] ciphertext_size The size of the created ciphertext
        @throws std::invalid_argument if ciphertext_size is less than 2
        @throws std::invalid_argument if noise_budget is not in the valid range
        */
  SimulatorCiphertext(AbstractCiphertextFactory &acf,
                      const seal::EncryptionParameters &parms,
                      int ciphertext_size,
                      int noise_budget);

  SimulatorCiphertext(const SimulatorCiphertext &other); // copy constructor

  SimulatorCiphertext(SimulatorCiphertext &&other) noexcept;  // copy assignment

  SimulatorCiphertext &operator=(const SimulatorCiphertext &other);  // move constructor

  SimulatorCiphertext &operator=(SimulatorCiphertext &&other) noexcept;  // move assignment

  /// Creates a new SimulatorCiphertext: a wrapper around the seal::Ciphertext class.
  /// \param simulatorFactory The factory that created this ciphertext.
  explicit SimulatorCiphertext(SimulatorCiphertextFactory &simulatorFactory);

  /// create seal ciphertext given encryption parameters and estimate its noise based on the given enc params
  /// TODO
  std::unique_ptr<AbstractCiphertext> createFresh(const seal::EncryptionParameters &param, int plain_max_coeff_count,
                                                  uint64_t plain_max_abs_value);

  /// estimates the noise heuristically of a multiplication op of two Seal ciphertexts
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
  double noiseBits() override;
  std::unique_ptr<AbstractCiphertext> clone() override;
  SimulatorCiphertextFactory &getFactory() override;
  const SimulatorCiphertextFactory &getFactory() const override;

  std::unique_ptr<AbstractCiphertext> relinearize();

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