#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_

#include <map>
#include <queue>
#include <utility>
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
 * A helper class to store the value of a variable and its associated datatype in the variableValues map.
 */
class VariableValue {
 private:
  Datatype datatype;
  std::unique_ptr<AbstractExpr> value;

 public:

  /// Destructor
  ~VariableValue() = default;

  /// Create directly
  VariableValue(Datatype dtype, AbstractExpr *varValue)
      : datatype(std::move(dtype)), value(varValue ? varValue->clone(false)->castTo<AbstractExpr>() : nullptr) {};

  /// Copy constructor
  VariableValue(const VariableValue &vv)
      : datatype(vv.datatype), value(vv.value ? vv.value->clone(false)->castTo<AbstractExpr>() : nullptr) {};

  /// Move constructor
  VariableValue(VariableValue &&vv) = default;

  /// Copy assignment
  VariableValue &operator=(const VariableValue &other) {
    datatype = other.datatype;
    value = std::unique_ptr<AbstractExpr>(other.value->clone(false)->castTo<AbstractExpr>());
    return *this;
  };

  /// Move assigment
  VariableValue &operator=(VariableValue &&other) = default;

  /// Creates a copy of the value!
  AbstractExpr *getValue() {
    return value ? value->clone(false)->castTo<AbstractExpr>() : nullptr;
  }

  Datatype getDatatype() {
    return datatype;
  }

  void setValue(AbstractExpr *val) {
    value = std::unique_ptr<AbstractExpr>(val->clone(false)->castTo<AbstractExpr>());
  }

  // Datatype is fixed, therefore no way to change this
};

typedef std::pair<std::string, Scope *> ScopedVariable;

/**
 * A helper class to store Mappings Between Variables and Scopes
 */
class VariableValuesMap {
 private:
  std::map<ScopedVariable, VariableValue *> variableValues;
 public:
  /// Create an empty map
  VariableValuesMap() = default;

  /// Destructor
  ~VariableValuesMap() = default;

  /// Create a map  directly
  explicit VariableValuesMap(std::map<ScopedVariable, VariableValue *> variableValues) : variableValues(std::move(
      variableValues)) {};

  /// Copy ctor (Deep-ish copy that clones AbstractExpr's stored as values)
  VariableValuesMap(const VariableValuesMap &other);

  /// Move ctor
  VariableValuesMap(VariableValuesMap &&other) noexcept = default;

  /// Copy assignment
  VariableValuesMap &operator=(const VariableValuesMap &other) {
    variableValues = other.variableValues;
    return *this;
  };

  /// Move assignment
  VariableValuesMap &operator=(VariableValuesMap &&other) {
    variableValues = std::move(other.variableValues);
    return *this;
  };

  /// Saves information about a declared variable. Must include the variable's identifier, the variable's datatype, and
  /// optionally also an initializer (or nullptr otherwise). Deep copies VariableValue
  void addDeclaredVariable(ScopedVariable scopedVariable, VariableValue *value);

  /// Saves information about a declared variable. Must include the variable's identifier, the variable's datatype, and
  /// optionally also an initializer (or nullptr otherwise).
  void addDeclaredVariable(ScopedVariable scopedVariable, VariableValue value);

  VariableValue *getVariableValue(ScopedVariable scopedVariable);

  void addVariable(ScopedVariable scopedVariable, VariableValue *value);

  void setVariableValue(ScopedVariable scopedVariable, VariableValue *value);

  /// Returns the current value of the variable identified by the given variableName. If there are multiple
  /// declarations within different scopes, returns the declaration that is closest to curScope.
  /// \param variableName The variable identifiers whose value should be retrieved.
  /// \return An AbstractExpr pointer of the variable's current value.
  AbstractExpr *getVariableValueDeclaredInThisOrOuterScope(std::string variableName, Scope *curScope);

  /// Returns the variable entry in variableValues that has the given variable identifier
  /// (variableName) and is closest from the current scope (curScope).
  /// \param variableName The variable identifiers whose variableValues entry should be retrieved.
  /// \return An iterator to the variableValues entry pointing to the variable whose declaratin is closest to the
  /// current scope.
  ScopedVariable getVariableEntryDeclaredInThisOrOuterScope(std::string variableName, Scope *curScope);

  //LAZY HACK
  const std::map<ScopedVariable, VariableValue *> &getMap() { return variableValues; };

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
