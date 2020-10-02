#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXT_H_

#include <memory>

class AbstractCiphertext {
 public:

  ///
  virtual ~AbstractCiphertext() = default;

  ///
  /// \param lhs
  /// \param operand
  /// \return
  virtual std::unique_ptr<AbstractCiphertext> multiply(AbstractCiphertext &operand) = 0;

  ///
  /// \param lhs
  /// \param operand
  virtual void multiplyInplace(AbstractCiphertext &operand) = 0;

  ///
  /// \param lhs
  /// \param operand
  /// \return
  virtual std::unique_ptr<AbstractCiphertext> add(AbstractCiphertext &operand) = 0;

  ///
  /// \param lhs
  /// \param operand
  virtual void addInplace(AbstractCiphertext &operand) = 0;

  /// \param lhs
  /// \param operand
  /// \return
  virtual std::unique_ptr<AbstractCiphertext> subtract(AbstractCiphertext &operand) = 0;

  ///
  /// \param lhs
  /// \param operand
  virtual void subtractInplace(AbstractCiphertext &operand) = 0;

  ///
  /// \param abstractCiphertext
  /// \param steps
  /// \return
  virtual std::unique_ptr<AbstractCiphertext> rotateRows(int steps) = 0;

  ///
  /// \param abstractCiphertext
  /// \param steps
  virtual void rotateRowsInplace(AbstractCiphertext &abstractCiphertext, int steps) = 0;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXT_H_
