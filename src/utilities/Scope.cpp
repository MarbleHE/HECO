#include "Scope.h"
#include <utility>
#include "VarDecl.h"

Scope::Scope(std::string scopeIdentifier, AbstractStatement *scopeOpener, Scope *outerScope)
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

Scope *Scope::getOrCreateInnerScope(const std::string &identifier, AbstractStatement *statement) {
  Scope *sc = findInnerScope(identifier);
  if (sc!=nullptr) {
    // return existing scope
    return sc;
  } else {
    // create and return new scope
    auto *newScope = new Scope(identifier, statement, this);
    this->innerScopes.insert({identifier, newScope});
    return newScope;
  }
}

void Scope::addStatement(AbstractStatement *absStatement) {
  this->scopeStatements.emplace_back(absStatement);
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

AbstractStatement *Scope::getScopeOpener() const {
  return scopeOpener;
}

