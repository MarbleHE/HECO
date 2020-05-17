#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_

#include <map>
#include <queue>
#include <vector>
#include <string>
#include <unordered_map>
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/Datatype.h"

class Scope {
 private:
  std::unordered_map<std::string, Scope *> innerScopes;
  std::string scopeIdentifier;
  AbstractNode *scopeOpener;
  Scope *outerScope;
  std::vector<AbstractStatement *> scopeStatements;

 public:
  Scope(std::string scopeIdentifier, AbstractNode *scopeOpener, Scope *outerScope);

  [[nodiscard]] Scope *getOuterScope() const;

  [[nodiscard]] const std::string &getScopeIdentifier() const;

  [[nodiscard]] const std::vector<AbstractStatement *> &getScopeStatements() const;

  [[nodiscard]] AbstractNode *getScopeOpener() const;

  Scope *getOrCreateInnerScope(const std::string &identifier, AbstractNode *statement);

  Scope *findInnerScope(const std::string &identifier);

  void addStatement(AbstractStatement *absStatement);

  void removeStatement(AbstractStatement *absStatement);

  /// Returns the nth last element of the scope statements.
  /// For example, n=1 prints the last element and n=2 the penultimate element.
  /// \param n The position of the element counted from back of the vector.
  /// \return The AbstractStatement at the n-th last position.
  AbstractStatement *getNthLastStatement(int n);

  AbstractStatement *getLastStatement();
};

/**
 * A helper struct to store the value of a variable and its associated datatype in the variableValues map.
 */
struct VariableValue {
  Datatype *datatype;
  AbstractExpr *value;

  VariableValue(Datatype *dtype, AbstractExpr *varValue) : datatype(dtype), value(varValue) {}

  // copy constructor
  VariableValue(const VariableValue &vv) {
    datatype = vv.datatype->clone(false)->castTo<Datatype>();
    value = (vv.value!=nullptr) ? vv.value->clone(false)->castTo<AbstractExpr>() : nullptr;
  }

  void setValue(AbstractExpr *val) {
    VariableValue::value = val;
  }
};
typedef std::pair<std::string, Scope *> ScopedVariable;
typedef std::map<ScopedVariable, VariableValue *> VariableValuesMapType;


#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
