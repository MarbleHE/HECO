#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARIABLEASSIGNMENT_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARIABLEASSIGNMENT_H_

#include <memory>
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/AbstractTarget.h"

// Forward Declaration of class Variable
class Variable;

class VariableAssignment : public AbstractStatement {
 private:
  /// Target (left hand side) of the assignment
  std::unique_ptr<AbstractTarget> target;

  /// Value (right hand side) of the assignment
  std::unique_ptr<AbstractExpression> value;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  VariableAssignment *clone_impl() const override;

 public:
  /// Destructor
  ~VariableAssignment() override;

  /// Create an assignment with a target (lhs) and value (rhs)
  /// \param target The variable to assign to (the assignment will take ownership)
  /// \param value  The value to assign (the assignment will take ownership)
  VariableAssignment(std::unique_ptr<AbstractTarget> target, std::unique_ptr<AbstractExpression> value);

  /// Copy constructor
  /// \param other VariableAssignment to copy
  VariableAssignment(const VariableAssignment &other);

  /// Move constructor
  /// \param other VariableAssignment to copy
  VariableAssignment(VariableAssignment &&other) noexcept ;

  /// Copy assignment
  /// \param other VariableAssignment to copy
  /// \return This object
  VariableAssignment &operator=(const VariableAssignment &other);

  /// Move assignment
  /// \param other VariableAssignment to move
  /// \return This object
  VariableAssignment &operator=(VariableAssignment &&other)  noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<VariableAssignment> clone() const;

  /// Does this assignment have its target set?
  /// \return true iff the assignment has the target set
  bool hasTarget() const;

  /// Does this assignment have its value set?
  /// \return true iff the assignment has the value set
  bool hasValue() const;

  /// Get (a reference to) the target (if it exists)
  /// \return A reference to the target
  /// \throws std::runtime_error if no target exists
  AbstractTarget &getTarget();

  /// Get (a const reference to) the target (if it exists)
  /// \return A reference to the target
  /// \throws std::runtime_error if no target exists
  const AbstractTarget &getTarget() const;

  /// Get (a reference to) the value (if it exists)
  /// \return A reference to the value variable
  /// \throws std::runtime_error if no value exists
  AbstractExpression &getValue();

  /// Get (a const reference to) the value (if it exists)
  /// \return A reference to the value variable
  /// \throws std::runtime_error if no value exists
  const AbstractExpression &getValue() const;

  /// Set the target to newTarget, taking ownership of newTarget
  /// This will delete the previous target!
  /// \param newTarget new target to set
  void setTarget(std::unique_ptr<AbstractTarget> newTarget);

  /// Set the value to newValue, taking ownership of newValue
  /// This will delete the previous value!
  /// \param newValue new value to set
  void setValue(std::unique_ptr<AbstractExpression> newValue);

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
class VariableAssignmentIteratorImpl : public PositionIteratorImpl<T, VariableAssignment> {
 public:
  // Inherit the constructor from the base class since it does everything we need
  using PositionIteratorImpl<T,VariableAssignment>::PositionIteratorImpl;

  T &operator*() override {
    switch (this->position) {
      case 0:
        if (this->node.hasTarget())
          return this->node.getTarget();
        else if (this->node.hasValue())
          return this->node.getValue();
        else
          throw std::runtime_error("Cannot dereference iterator since node has no children.");
      case 1:
        if (this->node.hasTarget())
          if (this->node.hasValue())
            return this->node.getValue();
        // If there is no target, then position 1 is past end even if value exists
        // If there is target, but no value, we're also past end, so just continue into default
      default:
        // calling dereference on higher elements is an error
        throw std::runtime_error("Trying to dereference iterator past end.");
    }
  }

  std::unique_ptr<BaseIteratorImpl<T>> clone() override {
    return std::make_unique<VariableAssignmentIteratorImpl>(this->node,this->position);
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARIABLEASSIGNMENT_H_
