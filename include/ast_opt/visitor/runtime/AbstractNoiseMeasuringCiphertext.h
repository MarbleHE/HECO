
#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTNOISEMEASURINGCIPHERTEXT_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTNOISEMEASURINGCIPHERTEXT_H_

#include "AbstractCiphertext.h"

class AbstractNoiseMeasuringCiphertext : public AbstractCiphertext {
 protected:
  /// Protected constructor: makes sure that class is abstract, i.e., cannot be instantiated.
  explicit AbstractNoiseMeasuringCiphertext(const std::reference_wrapper<const AbstractCiphertextFactory> acf)
      : AbstractCiphertext(acf) {};

 public:

  /// Return the current noise budget in the ciphertext, measured in bits
  /// \return An int measuring the number of bits in the noise budget.
  //virtual int64_t noiseBits() const = 0;

  /// Returns the current noise estimate
  /// \return current noise estimate
  virtual double getNoise() const = 0;

  /// Noise of a freshly encrypted ciphertext
  /// \return the noise budget in bits
  [[nodiscard]] virtual int64_t initialNoise() = 0;

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTNOISEMEASURINGCIPHERTEXT_H_
