#include "../../include/utilities/Scope.h"

#include <utility>

Scope::Scope(std::string scopeIdentifier, Scope *outerScope) :
        scopeIdentifier(std::move(scopeIdentifier)), outerScope(outerScope) {
}

Scope *Scope::getOuterScope() const {
    return outerScope;
}

const std::string &Scope::getScopeIdentifier() const {
    return scopeIdentifier;
}


Scope *Scope::findInnerScope(const std::string &identifier) {
    auto it = this->innerScopes.find(identifier);
    if (it != this->innerScopes.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

Scope *Scope::getOrCreateInnerScope(const std::string &identifier) {
    Scope *sc = findInnerScope(identifier);
    if (sc != nullptr) {
        // return existing scope
        return sc;
    } else {
        // create and return new scope
        auto *newScope = new Scope(identifier, this);
        this->innerScopes.insert({identifier, newScope});
        return newScope;
    }
}

void Scope::addStatement(AbstractStatement *absStatement) {
    this->scopeStatements.emplace_back(absStatement);
}

/// Prints the nth last element of the scope statements.
/// For example, n=1 prints the last element and n=2 the penultimate element.
/// \param n
/// \return
AbstractStatement *Scope::getNthLastStatement(int n) {
    if (n > scopeStatements.size()) {
        return nullptr;
    } else {
        auto it = scopeStatements.end();
        auto pv = std::prev(it, n);
        return *pv;
    }
}

const std::vector<AbstractStatement *> &Scope::getScopeStatements() const {
    return scopeStatements;
}

