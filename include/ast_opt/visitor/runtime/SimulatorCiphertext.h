#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SIMULATORCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SIMULATORCIPHERTEXT_H_

#include <memory>
#include "AbstractCiphertext.h"
#include "SimulatorCiphertextFactory.h"

// forward declarations
class SimulatorCiphertextFactory;

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

class SimulatorCiphertext : public AbstractCiphertext {
 private:
  SimulatorCiphertext(SimulatorCiphertextFactory &acf,
                      const seal::EncryptionParameters &parms,
                      int ciphertext_size,
                      int noise_budget);
  seal::Ciphertext ciphertext;
  uint64_t _noise; // current invariant noise
  double noise_budget = 0; // current noise budget
  std::unique_ptr<SimulatorCiphertext> clone_impl();

  // added by MW
  int64_t noise_;

  int64_t coeff_modulus_;

  int coeff_modulus_bit_count_ = 0;

  int ciphertext_size_ = 0;

  seal::EncryptionParameters parms_;


 public:
  ~SimulatorCiphertext() override = default;

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
                                                  uint64_t plain_max_abs_value) override;

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
  double noiseBits(int noise) override;
  std::unique_ptr<AbstractCiphertext> clone() override;
  SimulatorCiphertextFactory &getFactory() override;
  const SimulatorCiphertextFactory &getFactory() const override;


};

#endif
#endif