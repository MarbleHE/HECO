#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXT_H_

#include <memory>

class AbstractCiphertext {
 public:
  /// Default destructor.
  virtual ~AbstractCiphertext() = default;

  /// Multiplies this ciphertext with the given ciphertext and returns a new ciphertext with the result.
  /// \param operand The ciphertext to multiply this ciphertext with.
  /// \return A std::unique_ptr<AbstractCiphertext> containing the result ciphertext.
  virtual std::unique_ptr<AbstractCiphertext> multiply(AbstractCiphertext &operand) = 0;

  /// Multiplies this ciphertext with the given ciphertext by overwriting this ciphertext with the result.
  /// \param operand The ciphertext to multiply this ciphertext with.
  virtual void multiplyInplace(AbstractCiphertext &operand) = 0;

  /// Adds this ciphertext to the given ciphertext and returns a new ciphertext with the result.
  /// \param operand The ciphertext to add to this ciphertext.
  /// \return A std::unique_ptr<AbstractCiphertext> containing the result ciphertext.
  virtual std::unique_ptr<AbstractCiphertext> add(AbstractCiphertext &operand) = 0;

  /// Adds this ciphertext to the given ciphertext by overwriting this ciphertext with the result.
  /// \param operand The ciphertext to add to this ciphertext.
  virtual void addInplace(AbstractCiphertext &operand) = 0;

  /// Subtracts the given ciphertext from this ciphertext and returns a new ciphertext with the result.
  /// \param operand The ciphertext to add to this ciphertext.
  /// \return A std::unique_ptr<AbstractCiphertext> containing the result ciphertext.
  virtual std::unique_ptr<AbstractCiphertext> subtract(AbstractCiphertext &operand) = 0;

  /// Subtracts the given ciphertext from this given ciphertext by overwriting this ciphertext with the result.
  /// \param operand The ciphertext to subtract from this ciphertext.
  virtual void subtractInplace(AbstractCiphertext &operand) = 0;

  /// Cyclically rotates a copy of this ciphertext by the given number of steps and returns the rotated ciphertext.
  /// \param steps The number of steps a copy of this ciphertext should be rotated.
  /// \return A std::unique_ptr<AbstractCiphertext> containing the rotated ciphertext.
  virtual std::unique_ptr<AbstractCiphertext> rotateRows(int steps) = 0;

  /// Cyclically rotates this ciphertext by the given number of steps.
  /// \param steps The number of steps this ciphertext should be rotated.
  virtual void rotateRowsInplace(int steps) = 0;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXT_H_
