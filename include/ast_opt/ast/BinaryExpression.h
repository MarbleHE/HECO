#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_BinaryExpression_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_BinaryExpression_H_

#include <string>
#include <vector>

#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/utilities/Operator.h"

/// A BinaryExpression has two Operands (left and right) and an Operator
class BinaryExpression : public AbstractExpression {

 private:

  /// Left Hand Side
  std::unique_ptr<AbstractExpression> left;

  /// Operator (not part of node class hierarchy, i.e. not a child)
  Operator op;

  /// Right Hand Side
  std::unique_ptr<AbstractExpression> right;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  BinaryExpression *clone_impl() const override;
 public:
  /// Destructor
  ~BinaryExpression() override;

  /// Create a BinaryExpression with left hand side, operator and right hand side
  /// \param left Left Hand Side (the BinaryExpression will take ownership)
  /// \param operator Operator for this expression
  /// \param right  Right hand side (the BinaryExpression will take ownership)
  BinaryExpression(std::unique_ptr<AbstractExpression> left, Operator op, std::unique_ptr<AbstractExpression> right);

  /// Copy constructor
  /// \param other BinaryExpression to copy
  BinaryExpression(const BinaryExpression &other);

  /// Move constructor
  /// \param other BinaryExpression to copy
  BinaryExpression(BinaryExpression &&other) noexcept;

  /// Copy assignment
  /// \param other BinaryExpression to copy
  /// \return This object
  BinaryExpression &operator=(const BinaryExpression &other);

  /// Move assignment
  /// \param other BinaryExpression to move
  /// \return This object
  BinaryExpression &operator=(BinaryExpression &&other) noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<BinaryExpression> clone() const;

  /// Does this BinaryExpression have its left hand side set?
  /// \return true iff the assignment has the left hand side set
  bool hasLeft() const;

  /// Does this BinaryExpression have its operator set?
  /// Note: Currently always true
  bool hasOperator() const;

  /// Does this BinaryExpression have its right hand side set?
  /// \return true iff the assignment has the right hand side set
  bool hasRight() const;

  /// Get (a reference to) the left hand side (if it exists)
  /// \return A reference to the left hand side
  /// \throws std::runtime_error if no left hand side exists
  AbstractExpression &getLeft();

  /// Get (a const reference to) the left hand side (if it exists)
  /// \return A reference to the left hand side
  /// \throws std::runtime_error if no left hand side exists
  const AbstractExpression &getLeft() const;

  /// Get (a reference to) the Operator
  /// \return A reference to the operator variable
  Operator &getOperator();

  /// Get (a const reference to) the Operator
  /// \return A const reference to the operator variable
  const Operator &getOperator() const;

  /// Get (a reference to) the right hand side (if it exists)
  /// \return A reference to the right hand side
  /// \throws std::runtime_error if no right hand side exists
  AbstractExpression &getRight();

  /// Get (a const reference to) the right hand side (if it exists)
  /// \return A reference to the right hand side
  /// \throws std::runtime_error if no right hand side exists
  const AbstractExpression &getRight() const;

  /// Set the left hand side to newLeft, taking ownership of newLeft
  /// This will delete the previous left hand side!
  /// \param newLeft new left hand side to set
  void setLeft(std::unique_ptr<AbstractExpression> newLeft);

  /// Set the operator to newOperator
  /// \param newOperator new operator to set
  void setOperator(Operator newOperator);

  /// Set the right hand side to newRight, taking ownership of newRight
  /// This will delete the previous right hand side!
  /// \param newRight new right hand side to set
  void setRight(std::unique_ptr<AbstractExpression> newRight);

  ///////////////////////////////////////////////
  ////////// AbstractNode Interface /////////////
  ///////////////////////////////////////////////
  void accept(IVisitor &v) override;
  iterator begin() override;
  const_iterator begin() const override;
  iterator end() override;
  const_iterator end() const override;
  size_t countChildren() const override;
  nlohmann::json toJson() const override;
  std::string toString(bool printChildren) const override;
 protected:
  std::string getNodeType() const override;

};

// Designed to be instantiated only with T = (const) AbstractNode
template<typename T>
class BinaryExpressionIteratorImpl : public PositionIteratorImpl<T, BinaryExpression> {
 public:
  // Inherit the constructor from the base class since it does everything we need
  using PositionIteratorImpl<T, BinaryExpression>::PositionIteratorImpl;

  T &operator*() override {
    switch (this->position) {
      case 0:
        if (this->node.hasLeft())
          return this->node.getLeft();
        else if (this->node.hasRight())
          return this->node.getRight();
        else
          throw std::runtime_error("Cannot dereference iterator since node has no children.");
      case 1:
        if (this->node.hasLeft())
          if (this->node.hasRight())
            return this->node.getRight();
        // If there is no left hand side, then position 1 is past end even if value exists
        // If there is a left han side, but no right, we're also past end, so just continue into default
      default:
        // calling dereference on higher elements is an error
        throw std::runtime_error("Trying to dereference iterator past end.");
    }
  }

  std::unique_ptr<BaseIteratorImpl<T>> clone() override {
    return std::make_unique<BinaryExpressionIteratorImpl>(this->node, this->position);
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_BinaryExpression_H_
