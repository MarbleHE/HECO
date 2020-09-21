#include "ast_opt/utilities/Scope.h"

const ScopedIdentifier &Scope::resolveIdentifier(const std::string &id) const {
  // TODO: Replace dummy function body
  return **identifiers.begin();
}

Scope::Scope() {
  // TODO: Replace dummy function body
  identifiers.emplace(std::make_unique<ScopedIdentifier>());
}
Scope *ScopedIdentifier::getScope() {
  return scope;
}

const Scope *ScopedIdentifier::getScope() const {
  return scope;
}

void ScopedIdentifier::setScope(Scope *scope) {
  ScopedIdentifier::scope = scope;
}

const std::string &ScopedIdentifier::getId() const {
  return id;
}

std::string &ScopedIdentifier::getId() {
  return id;
}

void ScopedIdentifier::setId(const std::string &id) {
  ScopedIdentifier::id = id;
}
