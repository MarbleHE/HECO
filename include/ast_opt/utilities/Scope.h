#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_

#include <tuple>
#include <unordered_set>
#include "ast_opt/ast/AbstractNode.h"

//TODO: Add rule of five

// forward declarations
class Scope;

class ScopedIdentifier {
 private:
  /// (weak) pointer to the Scope this identifier belongs to
  Scope *scope;

  /// identifier (e.g., variable's name)
  std::string id;

 public:
  /// Creates a new ScopedIdentifier.
  /// \param scope The scope where the identifier was declared in.
  /// \param id The identifier that is defined in the given scope.
  ScopedIdentifier(Scope &scope, std::string id);

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

  bool operator==(const ScopedIdentifier &p) const;
};

class ScopedIdentifierHashFunction {
  /// This function must be passed to certain STL containers if they should contain GraphNodes, for example,
  /// std::unordered_set<std::reference_wrapper<GraphNode>, GraphNodeHashFunction> mySet.
 public:
  size_t operator()(const std::unique_ptr<ScopedIdentifier> &scopedIdentifier) const;

  size_t operator()(const ScopedIdentifier &scopedIdentifier) const;
};

class Scope {
 private:
  /// (Weak) pointer to the AST node that creates this scope
  AbstractNode *astNode;

  /// Set of identifiers declared in this scope
  std::unordered_set<std::unique_ptr<ScopedIdentifier>, ScopedIdentifierHashFunction> identifiers;

  /// Parent scope (if it exists)
  Scope *parent = nullptr;

  /// The scopes that are nested in this scope
  std::vector<std::unique_ptr<Scope>> nestedScopes;

 public:
  /// Destructor
  ~Scope() = default;

  /// Creates a new scope.
  /// \param abstractNode The node that creates this scope.
  explicit Scope(AbstractNode &abstractNode);

  /// Copy Constructor
  Scope(const Scope&) = delete;

  /// Move Constructor
  /// \param other the scope to move
  Scope(Scope&& other) = default;

  /// Copy Assignment
  Scope& operator=(const Scope&) = delete;

  /// Move Assignment
  Scope& operator=(Scope&&) noexcept = default;

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

  /// Adds a nested scope to this scope's nestedScopes vector.
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

  /// Determines the ScopedIdentifier of the given identifier.
  /// \param id The identifier for that the ScopedIdentifier should be determined.
  /// \return (A const reference) to the ScopedIdentifier object associated with the given identifier.
  ScopedIdentifier &resolveIdentifier(const std::string &id);

  /// Gets the nested scoped that is created by the given node. Note that this does NOT include deeper
  /// nested scopes. It only considers scopes that are directly (next level) nested into this scope.
  /// \param node The node that created the scope that is searched for.
  /// \return (A reference to) the scope created by the given node.
  Scope &getNestedScopeByCreator(AbstractNode &node);

  /// Gets the nested scoped that is created by the given node. Note that this does NOT include deeper
  /// nested scopes. It only considers scopes that are directly (next level) nested into this scope.
  /// \param node The node that created the scope that is searched for.
  /// \param node The node that created the scope that is searched for.
  /// \return (A const reference to) the scope created by the given node.
  const Scope &getNestedScopeByCreator(AbstractNode &node) const;

  /// Get Scope name
  /// \return the name of this scope (uniqueID of the associated AST node)
  std::string getScopeName() const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
