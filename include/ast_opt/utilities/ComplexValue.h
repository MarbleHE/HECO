#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_COMPLEXVALUE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_COMPLEXVALUE_H_
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/utilities/BatchingConstraint.h"

/// Represents a SIMD Ciphertext
class ComplexValue {
 private:
  /// Map representing the ctxt, mapping slot index to values
  std::map<int,std::reference_wrapper<AbstractExpression>> map;

 public:
  /// Create an empty ComplexValue
  ComplexValue() = default;

  ComplexValue(const ComplexValue& other) = default;

  ComplexValue(ComplexValue&& other) noexcept = default;

  ComplexValue& operator=(const ComplexValue& other) = default;

  ComplexValue& operator=(ComplexValue&& other) noexcept = default;

  /// Create a ComplexValue from a simple value (assign to all slots, i.e. slot -1)
  /// \param value Simple Value
  explicit ComplexValue(AbstractExpression& value);

  /// Create a ComplexValue from a simple value and a slot
  /// \param value Simple Value
  /// \param slot indicates target slot
  ComplexValue(AbstractExpression& value, int slot);

  void merge(ComplexValue& value);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_COMPLEXVALUE_H_
