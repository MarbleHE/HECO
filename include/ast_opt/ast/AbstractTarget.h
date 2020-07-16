#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTTARGET_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTTARGET_H_
#include "ast_opt/ast/AbstractExpression.h"

/// This class merely structures the inheritance hierarchy
/// It represents "things that can be assigned to", i.e. lvalues
class AbstractTarget : public AbstractExpression {
 public:
  /// Clones a node recursively, i.e., by including all of its children.
  /// Because return-type covariance does not work with smart pointers,
  /// derived classes are expected to introduce a std::unique_ptr<DerivedNode> clone() method that hides this (for use with derived class ptrs/refs)
  /// \return A clone of the node including clones of all of its children.
  inline std::unique_ptr<AbstractTarget> clone() const { /* NOLINT intentionally hiding */
    return std::unique_ptr<AbstractTarget>(clone_impl());
  }
 private:
  /// Refines return type to AbstractTarget
  AbstractTarget *clone_impl() const override = 0;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTEXPRESSION_H_
