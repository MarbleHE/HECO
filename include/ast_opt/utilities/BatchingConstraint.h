#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_BATCHINGCONSTRAINT_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_BATCHINGCONSTRAINT_H_
#include "ast_opt/utilities/Scope.h"
/// For now, we simply consider batching constraints to be a slot and a variable name valid in the local scope
class BatchingConstraint {
 private:
  int slot = -1;
  ScopedIdentifier identifier;
 public:
  BatchingConstraint() = default;

  BatchingConstraint(int slot, const ScopedIdentifier &identifier);

  [[nodiscard]] int getSlot() const;

  void setSlot(int slot);

  [[nodiscard]] const ScopedIdentifier &getIdentifier() const;

  void setIdentifier(const ScopedIdentifier &identifier);

  bool hasTargetSlot() const;
};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_BATCHINGCONSTRAINT_H_
