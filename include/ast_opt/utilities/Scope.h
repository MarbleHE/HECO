#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
#include <tuple>
#include <set>
#include "ast_opt/ast/AbstractNode.h"

//TODO: Finish defining interface, add rule of five & implementations

class ScopedIdentifier;

class Scope{
 private:
  /// (weak) pointer to the AST node that creates this scope
  AbstractNode *node = nullptr;

  /// Set of identifiers declared in this scope
  std::set<std::unique_ptr<ScopedIdentifier>> identifiers;

  /// Parent scope (if it exists)
  Scope *parent = nullptr;

 public:

  Scope();

  void addIdentifier(const std::string &id);

  bool identifierIsLocal(const std::string &id) const;

  bool identifierExists(const std::string &id) const;

  Scope &getParentScope();

  const Scope &getParentScope() const;

  [[nodiscard]] const ScopedIdentifier &resolveIdentifier(const std::string &id) const;
};

class ScopedIdentifier {
 private:
  /// (weak) pointer to the Scope this identifier belongs to
  Scope *scope;

  /// identifier
  std::string id;

 public:
  bool operator<(const ScopedIdentifier &fk) const {
    return id < fk.id;
  }

  Scope *getScope();

  const Scope *getScope() const;

  void setScope(Scope *scope);

  const std::string &getId() const;

  std::string &getId();

  void setId(const std::string &id);

};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_SCOPE_H_
