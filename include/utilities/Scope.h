#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_SCOPE_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_SCOPE_H_

#include <map>
#include <queue>
#include <vector>
#include <string>
#include "AbstractStatement.h"
#include <unordered_map>

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

  /// Returns the nth last element of the scope statements.
  /// For example, n=1 prints the last element and n=2 the penultimate element.
  /// \param n The position of the element counted from back of the vector.
  /// \return The AbstractStatement at the n-th last position.
  AbstractStatement *getNthLastStatement(int n);

  AbstractStatement *getLastStatement();
};

#endif //AST_OPTIMIZER_INCLUDE_UTILITIES_SCOPE_H_
