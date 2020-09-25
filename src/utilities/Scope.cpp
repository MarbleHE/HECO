#include <utility>
#include <iostream>

#include "ast_opt/utilities/Scope.h"

const ScopedIdentifier &Scope::resolveIdentifier(const std::string &id) const {
  // go through scopes, starting from the current scope and then walking up (parent nodes), by looking for the given
  // identifier
  const Scope *curScope = this;
  while (curScope!=nullptr) {
    auto it = std::find_if(curScope->identifiers.begin(),
                           curScope->identifiers.end(),
                           [&id](const auto &p) { return p->getId()==id; });
    if (it!=curScope->identifiers.end()) {
      return **it;
    }
    curScope = curScope->parent;
  }

  throw std::runtime_error("Identifier (" + id + ") cannot be resolved!");
}

void Scope::addIdentifier(const std::string &id) {
  identifiers.emplace(std::make_unique<ScopedIdentifier>(*this, id));
}

Scope &Scope::getParentScope() {
  return *parent;
}

const Scope &Scope::getParentScope() const {
  return *parent;
}

Scope *Scope::createNestedScope(Scope &parentScope, AbstractNode &scopeOpener) {
  auto scope = std::make_unique<Scope>(scopeOpener);
  Scope *scopePtr = scope.get();
  scope->setParent(&parentScope);
  scope->addNestedScope(std::move(scope));
  return scopePtr;
}

void Scope::setParent(Scope *parentScope) {
  Scope::parent = parentScope;
}

void Scope::addNestedScope(std::unique_ptr<Scope> &&scope) {
  nestedScopes.push_back(std::move(scope));
}

Scope::Scope(AbstractNode &abstractNode) : astNode(abstractNode) {}

bool Scope::identifierExists(const std::string &id) const {
  try {
    auto scopedIdentifier = resolveIdentifier(id);
  } catch (std::runtime_error &) {
    return false;
  }
  return true;
}

bool Scope::identifierIsLocal(const std::string &id) const {
  // go through all identifiers that are declared in this scope and check whether the identifier matches the given one
  return std::any_of(identifiers.begin(), identifiers.end(), [id](const auto &scopedIdentifier) {
    return scopedIdentifier->getId()==id;
  });
}

Scope &ScopedIdentifier::getScope() {
  return scope;
}

const Scope &ScopedIdentifier::getScope() const {
  return scope;
}

const std::string &ScopedIdentifier::getId() const {
  return id;
}

std::string &ScopedIdentifier::getId() {
  return id;
}

ScopedIdentifier::ScopedIdentifier(Scope &scope, std::string id) : scope(scope), id(std::move(id)) {}

bool ScopedIdentifier::operator==(const ScopedIdentifier &p) const {
  ScopedIdentifierHashFunction hashScopedIdentifier;
  return hashScopedIdentifier(*this)==hashScopedIdentifier(p);
}

size_t ScopedIdentifierHashFunction::operator()(const std::unique_ptr<ScopedIdentifier> &scopedIdentifier) const {
  return std::hash<Scope *>()(&scopedIdentifier->getScope()) ^ std::hash<std::string>()(scopedIdentifier->getId());
}

size_t ScopedIdentifierHashFunction::operator()(const ScopedIdentifier &scopedIdentifier) const {
  return std::hash<const Scope *>()(&scopedIdentifier.getScope()) ^ std::hash<std::string>()(scopedIdentifier.getId());
}
