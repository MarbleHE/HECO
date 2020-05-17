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

void VariableValuesMap::addDeclaredVariable(std::string varIdentifier,
                                            Datatype *dType,
                                            AbstractExpr *value,
                                            Scope *curScope) {
  // create a clone of the value to be added to variableValues, otherwise changing the original would also modify the
  // one stored in variableValues
  AbstractExpr *clonedValue = (value==nullptr) ? nullptr : value->clone(false)->castTo<AbstractExpr>();

  // store the value in the variableValues map for further use (e.g., substitution: replacing variable identifiers by
  // the value of the referenced variable)
  variableValues[std::pair(varIdentifier, curScope)] =
      new VariableValue(dType->clone(false)->castTo<Datatype>(), clonedValue);
}

VariableValuesMap VariableValuesMap::getChangedVariables(
    VariableValuesMapType variableValuesBeforeVisitingNode) {
  // the result list of changed variables with their respective value
  decltype(variableValuesBeforeVisitingNode) changedVariables;
  // Loop through all variables in the current variableValues and check for each if it changed.
  // It is important that we loop through variableValues instead of variableValuesBeforeVisitingNode because there may
  // be newly declared variables.
  for (auto &[varIdentifierScope, varValue] : variableValues) {
    // a variable is changed if it either was added (i.e., declaration of a new variable) or its value was changed

    // check if it is a newly declared variable or an existing one
    auto newDeclaredVariable = variableValuesBeforeVisitingNode.count(varIdentifierScope)==0;
    auto existingVariable = !newDeclaredVariable;

    // check if exactly one of both is a nullptr -> no need to compare their concrete value
    auto
        anyOfTwoIsNullptr = [&](std::pair<std::string, Scope *> varIdentifierScope, VariableValue *varValue) -> bool {
      return (variableValuesBeforeVisitingNode.at(varIdentifierScope)->value==nullptr)!=(varValue->value==nullptr);
    };

    // check if their value is unequal: compare the value of both but prior to that make sure that value is not nullptr
    auto valueIsUnequal = [&](std::pair<std::string, Scope *> varIdentifierScope, VariableValue *varValue) -> bool {
      return (variableValuesBeforeVisitingNode.at(varIdentifierScope)->value!=nullptr && varValue->value!=nullptr)
          && !variableValuesBeforeVisitingNode.at(varIdentifierScope)->value->isEqual(varValue->value);
    };

    if (newDeclaredVariable
        || (existingVariable
            && (anyOfTwoIsNullptr(varIdentifierScope, varValue) || valueIsUnequal(varIdentifierScope, varValue)))) {
      changedVariables.emplace(varIdentifierScope,
                               new VariableValue(varValue->datatype->clone(false)->castTo<Datatype>(),
                                                 varValue->value->clone(false)->castTo<AbstractExpr>()));
    }
  }
  return VariableValuesMap(changedVariables);
}

AbstractExpr *VariableValuesMap::getVariableValueDeclaredInThisOrOuterScope(std::string variableName, Scope *curScope) {
  return variableValues.find(getVariableEntryDeclaredInThisOrOuterScope(variableName, curScope))->second->value;
}

ScopedVariable VariableValuesMap::getVariableEntryDeclaredInThisOrOuterScope(std::string variableName,
                                                                             Scope *curScope) {
  // variables to store the iterator to the declaration that is closest in terms of scopes and the distance between the
  // current scope and the scope of the declaration (e.g., distance is zero iff. both are in the same scope)
  VariableValuesMapType::iterator closestDeclarationIterator;
  int closestDeclarationDistance = INT_MAX;

  // go through all variables declared yet
  for (auto it = variableValues.begin(); it!=variableValues.end(); ++it) {
    // if the variable's identifier ("name") does not match, continue iterating
    if ((*it).first.first!=variableName) continue;
    // check if this variable declaration is valid in the current scope: start from curScope and go the scope hierarchy
    // upwards until the current scope matches the scope of the declaration -> declaration is valid in current scope
    auto scope = curScope;
    int scopeDistance = 0;
    while (scope!=nullptr) {
      // check if the current scope and the scope of the declaration are the same
      if (scope==(*it).first.second) {
        // check if this found variable declaration has a lower scope distance
        if (scopeDistance < closestDeclarationDistance) {
          closestDeclarationDistance = scopeDistance;
          closestDeclarationIterator = it;
          break;
        }
      }
      // go to the next "higher" scope and increment the scope distance
      scope = scope->getOuterScope();
      scopeDistance++;
    }
  }
  // if the bestIteratorDistance has still its default value (INT_MAX), return the variableValue's end iterator,
  // otherwise return the variableValues entry (iterator) that is closest to the current scope
  auto returnIt = (closestDeclarationDistance==INT_MAX) ? variableValues.end() : closestDeclarationIterator;
  if (returnIt==variableValues.end()) {
    throw std::invalid_argument("No variable with identifier " + variableName + " exists in this scope.");
  } else {
    return returnIt->first;
  }
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
