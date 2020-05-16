#include <algorithm>
#include <utility>
#include "ast_opt/utilities/Scope.h"
#include "ast_opt/ast/VarDecl.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Block.h"

Scope::Scope(std::string scopeIdentifier, AbstractNode *scopeOpener, Scope *outerScope)
    : scopeIdentifier(std::move(scopeIdentifier)), scopeOpener(scopeOpener), outerScope(outerScope) {

}

Scope *Scope::getOuterScope() const {
  return outerScope;
}

const std::string &Scope::getScopeIdentifier() const {
  return scopeIdentifier;
}

Scope *Scope::findInnerScope(const std::string &identifier) {
  auto it = this->innerScopes.find(identifier);
  if (it!=this->innerScopes.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

Scope *Scope::getOrCreateInnerScope(const std::string &identifier, AbstractNode *statement) {
  Scope *sc = findInnerScope(identifier);
  if (sc!=nullptr) {
    // return existing scope
    return sc;
  } else {
    // create and return new scope
    Scope *newScope;
    // for loops (for, while), the scope opener should be the initializer
    if (auto forstmt = dynamic_cast<For *>(statement)) {
      newScope = new Scope(identifier, forstmt->getInitializer(), this);
    } else {
      newScope = new Scope(identifier, statement, this);
    }
    this->innerScopes.insert({identifier, newScope});
    return newScope;
  }
}

void Scope::addStatement(AbstractStatement *absStatement) {
  this->scopeStatements.emplace_back(absStatement);
}

void Scope::removeStatement(AbstractStatement *absStatement) {
  scopeStatements
      .erase(std::remove(scopeStatements.begin(), scopeStatements.end(), absStatement), scopeStatements.end());
}

AbstractStatement *Scope::getNthLastStatement(int n) {
  if (n > scopeStatements.size()) {
    return nullptr;
  } else {
    auto it = scopeStatements.end();
    auto pv = std::prev(it, n);
    return *pv;
  }
}

AbstractStatement *Scope::getLastStatement() {
  return getNthLastStatement(1);
}

const std::vector<AbstractStatement *> &Scope::getScopeStatements() const {
  return scopeStatements;
}

AbstractNode *Scope::getScopeOpener() const {
  return scopeOpener;
}

