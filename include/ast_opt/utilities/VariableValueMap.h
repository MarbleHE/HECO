#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VARIABLEVALUEMAP_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VARIABLEVALUEMAP_H_

#include <unordered_map>
#include <unordered_set>
#include "ast_opt/utilities/ComplexValue.h"
#include "ast_opt/utilities/Scope.h"

class VariableValueMap {
 private:

  std::unordered_map<ScopedIdentifier, ComplexValue&> map;

  /// Bool represents if it has been updated since last resetChangeFlags()
  std::unordered_set<ScopedIdentifier> changed;

  typedef std::pair<const ScopedIdentifier, std::tuple<bool,ComplexValue&>> value_type;

 public:

  VariableValueMap() = default;

  VariableValueMap(const std::unordered_map<ScopedIdentifier, ComplexValue &> &map);

  //TODO: RULE OF FIVE

  void add(ScopedIdentifier s, ComplexValue& cv);

  [[nodiscard]] const ComplexValue& get(const ScopedIdentifier& s) const;

  ComplexValue& take(const ScopedIdentifier& s);

  void update(const ScopedIdentifier& s, ComplexValue& cv) ;

  bool has(const ScopedIdentifier& s);

  void resetChangeFlags();

  [[nodiscard]] std::unordered_set<ScopedIdentifier> changedEntries() const;

  ComplexValue &getToModify(const ScopedIdentifier &s);
};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VARIABLEVALUEMAP_H_
