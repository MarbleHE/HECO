#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTSTATEMENT_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTSTATEMENT_H_
#include "abc/ast/AbstractNode.h"

/// This class merely structures the inheritance hierarchy
class AbstractStatement : public AbstractNode {
 public:
  /// Clones a node recursively, i.e., by including all of its children.
  /// Because return-type covariance does not work with smart pointers,
  /// derived classes are expected to introduce a std::unique_ptr<DerivedNode> clone() method that hides this (for use with derived class ptrs/refs)
  /// \return A clone of the node including clones of all of its children.
  inline std::unique_ptr<AbstractStatement> clone(AbstractNode* parent_) const {
    return std::unique_ptr<AbstractStatement>(clone_impl(parent_));
  }
 private:
  /// Refines return type to AbstractStatement
  AbstractStatement *clone_impl(AbstractNode* parent_) const override = 0;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTSTATEMENT_H_
