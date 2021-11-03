#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_RETURN_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_RETURN_H_

#include <string>
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/utilities/Datatype.h"


/// A Return statement has an (optional) value (AbstractExpression)
class Return : public AbstractStatement {
 private:
  /// (Optional) value
  std::unique_ptr<AbstractExpression> value;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  Return *clone_impl(AbstractNode* parent) const override;
 public:
  /// Destructor
  ~Return() override;

  /// Create a Return statement
  /// \param value  (Optional) The value to return (the Return will take ownership)
   Return(std::unique_ptr<AbstractExpression> value = nullptr);

  /// Copy constructor
  /// \param other Return to copy
  Return(const Return &other);

  /// Move constructor
  /// \param other Return to copy
  Return(Return &&other) noexcept ;

  /// Copy assignment
  /// \param other Return to copy
  /// \return This object
  Return &operator=(const Return &other);

  /// Move assignment
  /// \param other Return to move
  /// \return This object
  Return &operator=(Return &&other)  noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<Return> clone(AbstractNode* parent = nullptr) const;


  /// Does this return statement have its value set?
  /// \return true iff the assignment has the value set
  bool hasValue() const;

   /// Get (a reference to) the value (if it exists)
  /// \return A reference to the value variable
  /// \throws std::runtime_error if no value exists
  AbstractExpression &getValue();

  /// Get (a const reference to) the value (if it exists)
  /// \return A reference to the value variable
  /// \throws std::runtime_error if no value exists
  const AbstractExpression &getValue() const;

  /// Set the value to newValue, taking ownership of newValue
  /// This will delete the previous value!
  /// \param newValue new value to set
  void setValue(std::unique_ptr<AbstractExpression> newValue);

  /// Create a Return node from a nlohmann::json representation of this node.
  /// \return unique_ptr to a new Return node
  static std::unique_ptr<Return> fromJson(nlohmann::json j);

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
class ReturnIteratorImpl : public PositionIteratorImpl<T, Return> {
 public:
  // Inherit the constructor from the base class since it does everything we need
  using PositionIteratorImpl<T,Return>::PositionIteratorImpl;

  T &operator*() override {
    switch (this->position) {
      case 0:
        if (this->node.hasValue())
          return this->node.getValue();
        else
          throw std::runtime_error("Cannot dereference iterator since node has no children.");
      default:
        // calling dereference on higher elements is an error
        throw std::runtime_error("Trying to dereference iterator past end.");
    }
  }

  std::unique_ptr<BaseIteratorImpl<T>> clone() override {
    return std::make_unique<ReturnIteratorImpl>(this->node,this->position);
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_RETURN_H_
