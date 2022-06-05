#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_TernaryExpression_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_TernaryExpression_H_

#include <string>
#include "heco/ast/AbstractExpression.h"
#include "heco/ast/Block.h"

// forward declaration of custom iterator for TernaryExpression
template <typename T>
class TernaryExpressionIteratorImpl;

/// A Ternary Expression is essentially an If, but as an expression.
/// It has a condition, and a "then" and "else" expression (rather than a statement/branch)
class TernaryOperator : public AbstractExpression
{
private:
  /// Stores the condition
  std::unique_ptr<AbstractExpression> condition = nullptr;

  /// Stores the then expression
  std::unique_ptr<AbstractExpression> thenExpr = nullptr;

  /// Stores the (optional) else expression
  std::unique_ptr<AbstractExpression> elseExpr = nullptr;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  TernaryOperator *clone_impl(AbstractNode *parent) const override;

public:
  /// Destructor
  ~TernaryOperator() override;

  /// Create an TernaryExpression with condition, thenExpr and elseExpr
  /// \param condition Should evaluate to a boolean value
  /// \param thenExpr Value if condition evaluates to true
  /// \param elseExpr Value if condition evaluates to false
  TernaryOperator(std::unique_ptr<AbstractExpression> &&condition,
                  std::unique_ptr<AbstractExpression> &&thenExpr,
                  std::unique_ptr<AbstractExpression> &&elseExpr);

  /// Copy constructor
  /// \param other TernaryExpression to copy
  TernaryOperator(const TernaryOperator &other);

  /// Move constructor
  /// \param other TernaryExpression to copy
  TernaryOperator(TernaryOperator &&other) noexcept;

  /// Copy assignment
  /// \param other TernaryExpression to copy
  /// \return This object
  TernaryOperator &operator=(const TernaryOperator &other);

  /// Move assignment
  /// \param other TernaryExpression to move
  /// \return This object
  TernaryOperator &operator=(TernaryOperator &&other) noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<TernaryOperator> clone(AbstractNode *parent = nullptr) const;

  /// Does this TernaryExpression have a non-null condition?
  /// \return true iff the condition is non-null
  bool hasCondition() const;

  /// Does this TernaryExpression have a non-null thenExpr?
  /// \return true iff the thenExpr is non-null
  bool hasThenExpr() const;

  /// Does this TernaryExpression have a non-null elseExpr?
  /// \return true iff the elseExpr is non-null
  bool hasElseExpr() const;

  /// Get the condition, if it exists
  /// \return (a reference to) the condition
  /// \throws std::runtime_error if no condition exists
  AbstractExpression &getCondition();

  /// Get the condition, if it exists
  /// \return (a const reference to) the condition
  /// \throws std::runtime_error if no condition exists
  const AbstractExpression &getCondition() const;

  /// Get the thenExpr, if it exists
  /// \return (a reference to) the thenExpr
  /// \throws std::runtime_error if no thenExpr exists
  AbstractExpression &getThenExpr();

  /// Get the thenExpr, if it exists
  /// \return (a const reference to) the thenExpr
  /// \throws std::runtime_error if no thenExpr exists
  const AbstractExpression &getThenExpr() const;

  /// Get the elseExpr, if it exists
  /// \return (a reference to) the elseExpr
  /// \throws std::runtime_error if no elseExpr exists
  AbstractExpression &getElseExpr();

  /// Get the elseExpr, if it exists
  /// \return (a const reference to) the elseExpr
  /// \throws std::runtime_error if no elseExpr exists
  const AbstractExpression &getElseExpr() const;

  /// Set a new condition
  /// \param newCondition condition to set, TernaryExpression takes ownership
  void setCondition(std::unique_ptr<AbstractExpression> &&newCondition);

  /// Set a new thenExpr
  /// \param newthenExpr thenExpr to set, TernaryExpression takes ownership
  void setThenExpr(std::unique_ptr<AbstractExpression> &&newthenExpr);

  /// Set a new elseExpr
  /// \param newelseExpr elseExpr to set, TernaryExpression takes ownership
  void setElseExpr(std::unique_ptr<AbstractExpression> &&newelseExpr);

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
template <typename T>
class TernaryExpressionIteratorImpl : public PositionIteratorImpl<T, TernaryOperator>
{
public:
  // Inherit the constructor from the base class since it does everything we need
  using PositionIteratorImpl<T, TernaryOperator>::PositionIteratorImpl;

  T &operator*() override
  {
    switch (this->position)
    {
    case 0:
      if (this->node.hasCondition())
        return this->node.getCondition();
      else if (this->node.hasThenExpr())
        return this->node.getThenExpr();
      else if (this->node.hasElseExpr())
        return this->node.getElseExpr();
      else
        throw std::runtime_error("Cannot dereference iterator since node has no children.");
    case 1:
      if (this->node.hasCondition())
      {
        if (this->node.hasThenExpr())
          return this->node.getThenExpr();
        else if (this->node.hasElseExpr())
          return this->node.getElseExpr();
        // If there is a condition but no branches, position 1 is past end, continue to default
      }
      else if (this->node.hasElseExpr())
      {
        // If there is no condition, then position 1 must be the elseExpr. TernaryExpression not, continue into default
        return this->node.getElseExpr();
      }
      // fall through
    case 2:
      // Position 2 is only valid if there is a condition, thenExpr and elseExpr
      // If not, continue to default
      if (this->node.hasCondition())
        if (this->node.hasThenExpr())
          if (this->node.hasElseExpr())
            return this->node.getElseExpr();
      // fall through
    default:
      // calling dereference on higher elements is an error
      throw std::runtime_error("Trying to dereference iterator past end.");
    }
  }

  std::unique_ptr<BaseIteratorImpl<T>>
  clone()
      override
  {
    return std::make_unique<TernaryExpressionIteratorImpl>(this->node, this->position);
  }
};

#endif // AST_OPTIMIZER_INCLUDE_AST_OPT_AST_IF_H_
