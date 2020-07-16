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
#include "Datatype.h"

/// Helper function to get target identifiers from Decls and Assignments
std::string getVarTargetIdentifier(AbstractStatement const *stmt);

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
  std::unique_ptr<AbstractExpression> value;

 public:

  /// Destructor
  ~VariableValue() = default;

  /// Create directly
  VariableValue(Datatype dtype, AbstractExpression *varValue)
      : datatype(std::move(dtype)), value(varValue ? varValue->clone()->castTo<AbstractExpression>() : nullptr) {};

  /// Copy constructor
  VariableValue(const VariableValue &vv)
      : datatype(vv.datatype), value(vv.value ? vv.value->clone()->castTo<AbstractExpression>() : nullptr) {};

  /// Move constructor
  VariableValue(VariableValue &&vv)  noexcept : datatype(std::move(vv.datatype)), value(std::move(vv.value)) {};

  /// Copy assignment
  VariableValue &operator=(const VariableValue &other) {
    datatype = other.datatype;
    if (other.value) {
      value = std::unique_ptr<AbstractExpression>(other.value->clone()->castTo<AbstractExpression>());
    }
    return *this;
  };

  /// Move assigment
  VariableValue &operator=(VariableValue &&other) = default;

  /// Creates a copy of the value!
  AbstractExpression *getValue() const {
    return value ? value->clone()->castTo<AbstractExpression>() : nullptr;
  }

  Datatype getDatatype() {
    return datatype;
  }

  void setValue(AbstractExpression *val) {
    value = std::unique_ptr<AbstractExpression>(val->clone()->castTo<AbstractExpression>());
  }

  // Datatype is fixed, therefore no way to change this
};

class ScopedVariable {
 private:
  /// Variable Identifier
  std::string identifier;

  /// Scope (not lifecycle managed by this class)
  Scope *scope;

 public:
  /// Direct Constructor
  ScopedVariable(std::string identifier, Scope *scope) : identifier(std::move(identifier)), scope(scope) {};

  /// Identifier Getter
  [[nodiscard]] std::string getIdentifier() const {
    return identifier;
  }

  /// Scope Getter
  [[nodiscard]] Scope const *getScope() const {
    return scope;
  }

  /// Magic to support structured bindings, i.e. auto &[id, scope] : scopedvariable
  template<std::size_t N>
  decltype(auto) get() const {
    if constexpr (N==0) return identifier;
    else if constexpr (N==1) return getScope();
  }

};

/// Total ordering over ScopedVariables to allow std::map
bool operator<(const ScopedVariable &lhs, const ScopedVariable &rhs);

/// Equality for Scoped Variables, based on identifier and scope identifier
bool operator==(const ScopedVariable &lhs, const ScopedVariable &rhs);

/// Inequality for Scoped Variables, based on identifier and scope identifier
bool operator!=(const ScopedVariable &lhs, const ScopedVariable &rhs);

/// To support structured bindings, i.e. auto &[id,scope]: scopedvar
namespace std {
template<>
struct tuple_size<ScopedVariable>
    : std::integral_constant<std::size_t, 2> {
};
template<std::size_t N>
struct tuple_element<N, ScopedVariable> {
  using type = decltype(std::declval<ScopedVariable>().get<N>());
};
}

/**
 * A helper class to store Mappings Between Variables and Scopes
 */
class VariableValuesMap {
 private:
  std::map<ScopedVariable, VariableValue> variableValues;
 public:
  /// Create an empty map
  VariableValuesMap() = default;

  /// Destructor
  ~VariableValuesMap() = default;

  /// Create a map  directly
  explicit VariableValuesMap(std::map<ScopedVariable, VariableValue> variableValues) : variableValues(std::move(
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
  /// optionally also an initializer (or nullptr otherwise).
  void addDeclaredVariable(ScopedVariable scopedVariable, VariableValue value);

  /// Find the VariableValue associated with a (scoped) Variable
  /// \param scopedVariable The Variable for which we want to retrieve the Value
  /// \return A VariableValue wrapper. NOTE: The actual AbstractExpr* value might be nullptr if the variable is unknown
  /// \throws std::invalid_argument if scopedVariable does not exist in this VariableValuesMap
  VariableValue getVariableValue(ScopedVariable scopedVariable);

  /// Update an existing Variable's Value
  /// \param scopedVariable Variable to update NOTE: Variable must exist in this map already!
  /// \param value Value to write into map. Might contain AbstractExpr* that is nullptr
  /// \throws std::invalid_argument if scopedVariable does not exist in this VariableValuesMap
  void setVariableValue(ScopedVariable scopedVariable, VariableValue value);

  /// Returns the current value of the variable identified by the given variableName. If there are multiple
  /// declarations within different scopes, returns the declaration that is closest to curScope.
  /// \param variableName The variable identifiers whose value should be retrieved.
  /// \return An AbstractExpr pointer of the variable's current value.
  AbstractExpression *getVariableValueDeclaredInThisOrOuterScope(std::string variableName, Scope *curScope);

  /// Returns the variable entry in variableValues that has the given variable identifier
  /// (variableName) and is closest from the current scope (curScope).
  /// \param variableName The variable identifiers whose variableValues entry should be retrieved.
  /// \return An iterator to the variableValues entry pointing to the variable whose declaratin is closest to the
  /// current scope.
  ScopedVariable getVariableEntryDeclaredInThisOrOuterScope(std::string variableName, Scope *curScope);

  //LAZY HACK
  const std::map<ScopedVariable, VariableValue> &getMap() { return variableValues; };

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
