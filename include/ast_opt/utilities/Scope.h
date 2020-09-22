#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_

#include <tuple>
#include <set>
#include "ast_opt/ast/AbstractNode.h"

//TODO: Finish defining interface, add rule of five & implementations

class ScopedIdentifier;

class Scope {
 private:
  /// (Weak) pointer to the AST node that creates this scope
  AbstractNode &astNode;

  /// Set of identifiers declared in this scope
  std::set<std::unique_ptr<ScopedIdentifier>> identifiers;

  /// Parent scope (if it exists)
  Scope *parent = nullptr;

  /// The scopes that are nested in this scope
  std::vector<std::unique_ptr<Scope>> nestedScopes;

 public:
  /// Creates a new scope.
  /// \param abstractNode The node that creates this scope.
  explicit Scope(AbstractNode &abstractNode);

  /// Adds an declared identifier (e.g., variable) to this scope.
  /// \param id The identifier to be added to this scope.
  void addIdentifier(const std::string &id);

  /// Checks whether the given identifier is local, i.e., declared in this scope and not in any parent scope.
  /// \param id The identifier to be checked.
  /// \return True iff the given identifier id is defined in this scope.
  [[nodiscard]] bool identifierIsLocal(const std::string &id) const;

  /// Checks whether the given identifier id exists in this or any parent scope.
  /// \param id The identifier to be checked.
  /// \return True iff the given identifier is defined in this or any parent scope.
  [[nodiscard]] bool identifierExists(const std::string &id) const;

  /// Sets the parent scope of this scope.
  /// \param parentScope The scope to be set as parent of this scope.
  void setParent(Scope *parentScope);

  /// Creates a nested scope within this scope.
  /// \param scope The scope to add as nested scope in this scope.
  void addNestedScope(std::unique_ptr<Scope> &&scope);

  /// Gets the parent scope of this scope.
  /// \return (A reference to) the parent scope of this scope.
  Scope &getParentScope();

  /// Gets the parent scope of this scope.
  /// \return (A const reference to) the parent scope of this scope.
  [[nodiscard]] const Scope &getParentScope() const;

  /// Creates a nested scope with the given parentScope as parent and the given AbstractNode scopeOpener as the AST node
  /// that creates this scope.
  /// \param parentScope The parent scope of the scope to be created.
  /// \param scopeOpener The AST node that creates the nested scope.
  /// \return (A weak pointer) to the nested scope that is created.
  static Scope *createNestedScope(Scope &parentScope, AbstractNode &scopeOpener);

  /// Determines the ScopedIdentifier of the given identifier.
  /// \param id The identifier for that the ScopedIdentifier should be determined.
  /// \return (A const reference) to the ScopedIdentifier object associated with the given identifier.
  [[nodiscard]] const ScopedIdentifier &resolveIdentifier(const std::string &id) const;
};

class ScopedIdentifier {
 private:
  /// (weak) pointer to the Scope this identifier belongs to
  Scope &scope;

  /// identifier (e.g., variable's name)
  std::string id;

 public:
  /// Creates a new ScopedIdentifier.
  /// \param scope The scope where the identifier was declared in.
  /// \param id The identifier that is defined in the given scope.
  ScopedIdentifier(Scope &scope, std::string id);

  bool operator<(const ScopedIdentifier &fk) const {
    return id < fk.id;
  }

  /// Gets the scope associated with this ScopedIdentifier.
  /// \return (A reference to) the scope of this ScopedIdentifier.
  Scope &getScope();

  /// Gets the scope associated with this ScopedIdentifier.
  /// \return (A const reference to) the scope of this ScopedIdentifier.
  [[nodiscard]] const Scope &getScope() const;

  /// Gets the identifier associated with this ScopedIdentifier.
  /// \return (A const string reference to) the identifier of this ScopedIdentifier.
  [[nodiscard]] const std::string &getId() const;

  /// Gets the identifier associated with this ScopedIdentifier.
  /// \return (A string reference to) the identifier of this ScopedIdentifier.
  std::string &getId();
};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
