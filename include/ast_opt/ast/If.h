#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_IF_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_IF_H_

#include <string>
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/Block.h"

// forward declaration of custom iterator for If
template<typename T>
class IfIteratorImpl;

class If : public AbstractStatement {
 private:
  /// Stores the condition
  std::unique_ptr<AbstractExpression> condition = nullptr;

  /// Stores the then branch
  std::unique_ptr<Block> thenBranch = nullptr;

  /// Stores the (optional) else branch
  std::unique_ptr<Block> elseBranch = nullptr;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  If *clone_impl() const override;

 public:
  ///Destructor
  ~If() override;

  /// Create an If statement with condition, thenBranch and (optionally) elseBranch
  /// \param condition Should evaluate to a boolean value
  /// \param thenBranch Statements to be executed if condition evaluates to true
  /// \param elseBranch Statements to be executed if condition evaluates to false
  If(std::unique_ptr<AbstractExpression> &&condition,
     std::unique_ptr<Block> &&thenBranch,
     std::unique_ptr<Block> &&elseBranch = nullptr);

  /// Copy constructor
  /// \param other If to copy
  If(const If &other);

  /// Move constructor
  /// \param other If to copy
  If(If &&other) noexcept;

  /// Copy assignment
  /// \param other If to copy
  /// \return This object
  If &operator=(const If &other);

  /// Move assignment
  /// \param other If to move
  /// \return This object
  If &operator=(If &&other) noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<If> clone() const;

  /// Does this If have a non-null condition?
  /// \return true iff the condition is non-null
  bool hasCondition() const;

  /// Does this If have a non-null thenBranch?
  /// \return true iff the thenBranch is non-null
  bool hasThenBranch() const;

  /// Does this If have a non-null elseBranch?
  /// \return true iff the elseBranch is non-null
  bool hasElseBranch() const;

  /// Get the condition, if it exists
  /// \return (a reference to) the condition
  /// \throws std::runtime_error if no condition exists
  AbstractExpression &getCondition();

  /// Get the condition, if it exists
  /// \return (a const reference to) the condition
  /// \throws std::runtime_error if no condition exists
  const AbstractExpression &getCondition() const;

  /// Get the thenBranch, if it exists
  /// \return (a reference to) the thenBranch
  /// \throws std::runtime_error if no thenBranch exists
  Block &getThenBranch();

  /// Get the thenBranch, if it exists
  /// \return (a const reference to) the thenBranch
  /// \throws std::runtime_error if no thenBranch exists
  const Block &getThenBranch() const;

  /// Get the elseBranch, if it exists
  /// \return (a reference to) the elseBranch
  /// \throws std::runtime_error if no elseBranch exists
  Block &getElseBranch();

  /// Get the elseBranch, if it exists
  /// \return (a const reference to) the elseBranch
  /// \throws std::runtime_error if no elseBranch exists
  const Block &getElseBranch() const;

  /// Set a new condition
  /// \param newCondition condition to set, If takes ownership
  void setCondition(std::unique_ptr<AbstractExpression> &&newCondition);

  /// Set a new thenBranch
  /// \param newThenBranch thenBranch to set, If takes ownership
  void setThenBranch(std::unique_ptr<Block> &&newThenBranch);

  /// Set a new elseBranch
  /// \param newElseBranch elseBranch to set, If takes ownership
  void setElseBranch(std::unique_ptr<Block> &&newElseBranch);

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
class IfIteratorImpl : public PositionIteratorImpl<T, If> {
 public:
  // Inherit the constructor from the base class since it does everything we need
  using PositionIteratorImpl<T, If>::PositionIteratorImpl;

  T &operator*() override {
    switch (this->position) {
      case 0:
        if (this->node.hasCondition())
          return this->node.getCondition();
        else if (this->node.hasThenBranch())
          return this->node.getThenBranch();
        else if (this->node.hasElseBranch())
          return this->node.getElseBranch();
        else
          throw std::runtime_error("Cannot dereference iterator since node has no children.");
      case 1:
        if (this->node.hasCondition()) {
          if (this->node.hasThenBranch())
            return this->node.getThenBranch();
          else if (this->node.hasElseBranch())
            return this->node.getElseBranch();
          // If there is a condition but no branches, position 1 is past end, continue to default
        } else if (this->node.hasElseBranch()) {
          // If there is no condition, then position 1 must be the elseBranch. If not, continue into default
          return this->node.getElseBranch();
        }
      case 2:
        // Position 2 is only valid if there is a condition, thenBranch and elseBranch
        // If not, continue to default
        if (this->node.hasCondition())
          if (this->node.hasThenBranch())
            if (this->node.hasElseBranch())
              return this->node.getElseBranch();
      default:
        // calling dereference on higher elements is an error
        throw std::runtime_error("Trying to dereference iterator past end.");
    }
  }

  std::unique_ptr<BaseIteratorImpl<T>>
  clone()
  override {
    return std::make_unique<IfIteratorImpl>(this->node, this->position);
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_IF_H_
