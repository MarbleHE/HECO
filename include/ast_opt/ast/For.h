#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FOR_H_

#include <string>
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/Block.h"

class For : public AbstractStatement {
 private:
  /// Stores the the initializer
  std::unique_ptr<Block> initializer = nullptr;

  /// Stores the condition
  std::unique_ptr<AbstractExpression> condition = nullptr;

  /// Stores the update part
  std::unique_ptr<Block> update = nullptr;

  /// Stores the body
  std::unique_ptr<Block> body = nullptr;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  For *clone_impl(AbstractNode* parent_) const override;

 public:
  ///Destructor
  ~For() override;

  /// Create an For statement with initializer, condition, update and body
  /// \param initializer Any Block of statements, usually setting up loop variables (e.g. int i = 0;)
  /// \param condition Should evaluate to a boolean value. Loop continues until condition is false
  /// \param update Any Block of statements to be run after each loop iteration (e.g. ++i)
  /// \param body Any Block of statements, to be executed in each loop iteration
  For(std::unique_ptr<Block> initializer,
      std::unique_ptr<AbstractExpression> condition,
      std::unique_ptr<Block> update,
      std::unique_ptr<Block> body);

  /// Copy constructor
  /// \param other For to copy
  For(const For &other);

  /// Move constructor
  /// \param other For to copy
  For(For &&other) noexcept;

  /// Copy assignment
  /// \param other For to copy
  /// \return This object
  For &operator=(const For &other);

  /// Move assignment
  /// \param other For to move
  /// \return This object
  For &operator=(For &&other) noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<For> clone(AbstractNode* parent = nullptr) const;

  /// Does this For have a non-null initializer?
  /// \return true iff the initializer is non-null
  bool hasInitializer() const;

  /// Does this For have a non-null condition?
  /// \return true iff the condition is non-null
  bool hasCondition() const;

  /// Does this For have a non-null update block?
  /// \return true iff the update block is non-null
  bool hasUpdate() const;

  /// Does this For have a non-null body?
  /// \return true iff the body is non-null
  bool hasBody() const;

  /// Get the initializer, if it exists
  /// \return (a reference to) the initializer
  /// \throws std::runtime_error if no initializer exists
  Block &getInitializer();

  /// Get the initializer, if it exists
  /// \return (a const reference to) the initializer
  /// \throws std::runtime_error if no initializer exists
  const Block &getInitializer() const;

  /// Get the condition, if it exists
  /// \return (a reference to) the condition
  /// \throws std::runtime_error if no condition exists
  AbstractExpression &getCondition();

  /// Get the condition, if it exists
  /// \return (a const reference to) the condition
  /// \throws std::runtime_error if no condition exists
  const AbstractExpression &getCondition() const;

  /// Get the update Block, if it exists
  /// \return (a reference to) the update Block
  /// \throws std::runtime_error if no update Block exists
  Block &getUpdate();

  /// Get the update Block, if it exists
  /// \return (a const reference to) the update Block
  /// \throws std::runtime_error if no update Block exists
  const Block &getUpdate() const;

  /// Get the body, if it exists
  /// \return (a reference to) the body
  /// \throws std::runtime_error if no body exists
  Block &getBody();

  /// Get the body, if it exists
  /// \return (a const reference to) the body
  /// \throws std::runtime_error if no body exists
  const Block &getBody() const;

  /// Set a new initializer
  /// \param newInitializer initializer to set, For takes ownership
  void setInitializer(std::unique_ptr<Block> newInitializer);

  /// Set a new condition
  /// \param newCondition condition to set, For takes ownership
  void setCondition(std::unique_ptr<AbstractExpression> newCondition);

  /// Set a new update Block
  /// \param newUpdate update Block to set, For takes ownership
  void setUpdate(std::unique_ptr<Block> newUpdate);

  /// Set a new body
  /// \param newBody body to set, For takes ownership
  void setBody(std::unique_ptr<Block> newBody);

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
class ForIteratorImpl : public PositionIteratorImpl<T, For> {
 public:
  // Inherit the constructor from the base class since it does everything we need
  using PositionIteratorImpl<T, For>::PositionIteratorImpl;

  T &operator*() override {
    switch (this->position) {
      case 0:
        if (this->node.hasInitializer())
          return this->node.getInitializer();
        else if (this->node.hasCondition())
          return this->node.getCondition();
        else if (this->node.hasUpdate())
          return this->node.getUpdate();
        else if (this->node.hasBody())
          return this->node.getBody();
        else
          throw std::runtime_error("Cannot dereference iterator since node has no children.");
      case 1:
        if (this->node.hasInitializer()) {
          if (this->node.hasCondition())
            return this->node.getCondition();
          else if (this->node.hasUpdate())
            return this->node.getUpdate();
          else if (this->node.hasBody())
            return this->node.getBody();
          // If there is a initializer but nothing else, position 1 is past end
        } else if (this->node.hasCondition()) {
          if (this->node.hasUpdate())
            return this->node.getUpdate();
          else if (this->node.hasBody())
            return this->node.getBody();
        } else if (this->node.hasUpdate()) {
          if (this->node.hasBody())
            return this->node.getBody();
        } // if there is only a body, position 1 is already past end
        throw std::runtime_error("Trying to dereference iterator past end.");
      case 2:
        if (this->node.hasInitializer()) {//0
          if (this->node.hasCondition()) { //1
            if (this->node.hasUpdate())
              return this->node.getUpdate();
            else if (this->node.hasBody())
              return this->node.getBody();
          } else if (this->node.hasUpdate()) {//1
            if (this->node.hasBody())
              return this->node.getBody();
          }
        } else if (this->node.hasCondition()) { //0
          if (this->node.hasUpdate()) // 1
            if (this->node.hasBody())
              return this->node.getBody();
        }
        throw std::runtime_error("Trying to dereference iterator past end.");
      case 3:
        // Position 3 is only valid if there is everything
        // If not, continue to default
        if (this->node.hasInitializer())
          if (this->node.hasCondition())
            if (this->node.hasUpdate())
              if (this->node.hasBody())
                return this->node.getBody();
      default:
        // calling dereference on higher elements is an error
        throw std::runtime_error("Trying to dereference iterator past end.");
    }
  }

  std::unique_ptr<BaseIteratorImpl<T>>
  clone()
  override {
    return std::make_unique<ForIteratorImpl>(this->node, this->position);
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FOR_H_
