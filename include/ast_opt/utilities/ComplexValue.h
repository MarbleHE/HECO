#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_COMPLEXVALUE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_COMPLEXVALUE_H_
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/utilities/BatchingConstraint.h"

class ComplexValue {
 private:
  BatchingConstraint batchingConstraint;

 public:
  /// Create a ComplexValue from a simple value and a BatchingConstraint
  /// \param value Simple Value
  /// \param batchingConstraint Wrapper around variable name and slot containing the value
  explicit ComplexValue(AbstractExpression& value);

  /// Get the value's bathing constraints (i.e. where it lives)
  BatchingConstraint& getBatchingConstraint();

  void merge(ComplexValue value);

  std::vector<std::unique_ptr<AbstractStatement>> statementsToExecutePlan();
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_COMPLEXVALUE_H_
