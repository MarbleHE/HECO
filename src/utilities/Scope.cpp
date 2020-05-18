#include <algorithm>
#include <utility>
#include <climits>
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

void VariableValuesMap::addDeclaredVariable(ScopedVariable scopedVariable, VariableValue *value) {
  if (variableValues.find(scopedVariable)!=variableValues.end()) {
    throw std::invalid_argument("Variable " + scopedVariable.first + " already exists in this scope!");
  } else {
    auto value_copy = value ? new VariableValue(*value) : nullptr;
    variableValues.insert_or_assign(scopedVariable, value_copy);
  }
}

void VariableValuesMap::addDeclaredVariable(ScopedVariable scopedVariable, VariableValue value) {
  if (variableValues.find(scopedVariable)!=variableValues.end()) {
    throw std::invalid_argument("Variable " + scopedVariable.first + " already exists in this scope!");
  } else {
    auto value_copy = new VariableValue(value);
    variableValues.insert_or_assign(scopedVariable, value_copy);
  }
}

AbstractExpr *VariableValuesMap::getVariableValueDeclaredInThisOrOuterScope(std::string variableName, Scope *curScope) {
  auto vv = variableValues.find(getVariableEntryDeclaredInThisOrOuterScope(variableName, curScope));
  if (vv==variableValues.end()) {
    throw std::invalid_argument("Variable " + variableName + " does not exist in this scope.");
  } else {
    return vv->second ? vv->second->getValue() : nullptr;
  }

}

ScopedVariable VariableValuesMap::getVariableEntryDeclaredInThisOrOuterScope(std::string variableName,
                                                                             Scope *curScope) {

  //  ALTERNATIVE IMPLEMENTATION FOR SIMPLER DEBUGGING:
  //  Since we don't have the VariableValuesMap sorted by scopes, we're iterating over it for each scope
  //  This is somewhat inefficient in deep call stacks
  auto scope = curScope;
  while (scope!=nullptr) {
    for (auto &[sv, vv] : variableValues) {
      if (sv.second==scope && sv.first==variableName) {
        return sv;
      }
    }
    scope = scope->getOuterScope();
  }
  throw std::invalid_argument("No variable with identifier " + variableName + " exists in this scope.");

//  // variables to store the iterator to the declaration that is closest in terms of scopes and the distance between the
//  // current scope and the scope of the declaration (e.g., distance is zero iff. both are in the same scope)
//  std::map<ScopedVariable, VariableValue *>::iterator closestDeclarationIterator;
//  int closestDeclarationDistance = INT_MAX;
//
//  // go through all variables declared yet
//  for (auto it = variableValues.begin(); it!=variableValues.end(); ++it) {
//    // if the variable's identifier ("name") does not match, continue iterating
//    if ((*it).first.first!=variableName) continue;
//    // check if this variable declaration is valid in the current scope: start from curScope and go the scope hierarchy
//    // upwards until the current scope matches the scope of the declaration -> declaration is valid in current scope
//    auto scope = curScope;
//    int scopeDistance = 0;
//    while (scope!=nullptr) {
//      // check if the current scope and the scope of the declaration are the same
//      if (scope==(*it).first.second) {
//        // check if this found variable declaration has a lower scope distance
//        if (scopeDistance < closestDeclarationDistance) {
//          closestDeclarationDistance = scopeDistance;
//          closestDeclarationIterator = it;
//          break;
//        }
//      }
//      // go to the next "higher" scope and increment the scope distance
//      scope = scope->getOuterScope();
//      scopeDistance++;
//    }
//  }
//  // if the bestIteratorDistance has still its default value (INT_MAX), return the variableValue's end iterator,
//  // otherwise return the variableValues entry (iterator) that is closest to the current scope
//  auto returnIt = (closestDeclarationDistance==INT_MAX) ? variableValues.end() : closestDeclarationIterator;
//  if (returnIt==variableValues.end()) {
//    throw std::invalid_argument("No variable with identifier " + variableName + " exists in this scope.");
//  } else {
//    return returnIt->first;
//  }
}
VariableValue *VariableValuesMap::getVariableValue(ScopedVariable scopedVariable) {
  auto it = variableValues.find(scopedVariable);
  if (it==variableValues.end()) {
    throw std::invalid_argument("Variable " + scopedVariable.first + " not found.");
  } else {
    return it->second;
  }
}
void VariableValuesMap::addVariable(ScopedVariable scopedVariable, VariableValue *value) {
  auto it = variableValues.find(scopedVariable);
  if (it!=variableValues.end()) {
    throw std::invalid_argument("Variable " + scopedVariable.first + " already exists.");
  } else {
    variableValues.insert_or_assign(scopedVariable, value);
  }
}
void VariableValuesMap::setVariableValue(ScopedVariable scopedVariable, VariableValue *value) {
  auto it = variableValues.find(scopedVariable);
  if (it==variableValues.end()) {
    throw std::invalid_argument("Variable " + scopedVariable.first + " not found.");
  } else {
    it->second = value;
  }
}
VariableValuesMap::VariableValuesMap(const VariableValuesMap &other) {
  // call the object's copy constructor for each VariableValue
  for (auto &[k, v] : other.variableValues) {
    variableValues.insert_or_assign(k, new VariableValue(*v));
  }
}
