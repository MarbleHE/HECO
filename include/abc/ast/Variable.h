#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARIABLE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARIABLE_H_

#include <map>
#include <string>
#include <vector>
#include "abc/ast/AbstractTarget.h"

/// Named lvalue (any string)
class Variable : public AbstractTarget {
 private:
  /// Name of this variable
  std::string identifier;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  Variable *clone_impl(AbstractNode* parent) const override;

 public:
  /// Destructor
  ~Variable() override;

  /// Create a variable with name identifier
  /// \param identifier Variable name, can be any valid string
  explicit Variable(std::string identifier);

  /// Copy constructor
  /// \param other Variable to copy
  Variable(const Variable &other);

  /// Move constructor
  /// \param other Variable to copy
  Variable(Variable &&other) noexcept ;

  /// Copy assignment
  /// \param other Variable to copy
  /// \return This object
  Variable &operator=(const Variable &other);

  /// Move assignment
  /// \param other Variable to move
  /// \return This object
  Variable &operator=(Variable &&other)  noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<Variable> clone(AbstractNode* parent = nullptr) const;

  [[nodiscard]] std::string getIdentifier() const;

  /// Create a Variable node from a nlohmann::json representation of this node.
  /// \return unique_ptr to a new Variable node
  static std::unique_ptr<Variable> fromJson(nlohmann::json j);

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

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARIABLE_H_
