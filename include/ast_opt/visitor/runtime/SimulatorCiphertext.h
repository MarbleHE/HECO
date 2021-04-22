#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SIMULATORCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_SIMULATORCIPHERTEXT_H_

#include "AbstractCiphertext.h"
#include "SimulatorCiphertextFactory.h"

// forward declarations
class SimulatorCiphertextFactory;

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

class SimulatorCiphertext : public AbstractCiphertext {
 private:
  seal::Ciphertext ciphertext;
  double noise_budget = 0;
  std::unique_ptr<SimulatorCiphertext> clone_impl();
 public:
  ~SimulatorCiphertext() override = default;
  int64_t initialNoise(AbstractCiphertext &operand) override;
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


};

#endif
#endif