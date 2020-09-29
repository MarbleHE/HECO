#include "ast_opt/ast/AbstractTarget.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/Vectorizer.h"

////////////////////////////////////////////
////        BatchingConstraint          ////
////////////////////////////////////////////
BatchingConstraint::BatchingConstraint(int slot, const ScopedIdentifier &identifier)
    : slot(slot), identifier(identifier) {}

int BatchingConstraint::getSlot() const {
  return slot;
}
void BatchingConstraint::setSlot(int slot_) {
  slot = slot_;
}
const ScopedIdentifier &BatchingConstraint::getIdentifier() const {
  return identifier;
}
void BatchingConstraint::setIdentifier(const ScopedIdentifier &identifier_) {
  identifier = identifier_;
}
bool BatchingConstraint::hasTargetSlot() const {
  return getSlot()!=-1;
}

////////////////////////////////////////////
////           ComplexValue             ////
////////////////////////////////////////////
ComplexValue::ComplexValue(AbstractExpression &) {
  //TODO: Implement ComplexValue Ctor
}

BatchingConstraint &ComplexValue::getBatchingConstraint() {
  //TODO: Implement ComplexValue::getBatchingConstraint
  return batchingConstraint;
}
void ComplexValue::merge(ComplexValue value) {
  //TODO: Implement
}
std::vector<std::unique_ptr<AbstractStatement>> ComplexValue::statementsToExecutePlan() {
  //TODO: Implement
  return {};
}

////////////////////////////////////////////
////         VariableValueMap           ////
////////////////////////////////////////////
void VariableValueMap::add(ScopedIdentifier s, ComplexValue &cv) {
  map.insert({s, cv});
  changed.insert(s);
}

const ComplexValue &VariableValueMap::get(const ScopedIdentifier &s) const {
  return map.find(s)->second;
}

ComplexValue &VariableValueMap::getToModify(const ScopedIdentifier &s) {
  changed.insert(s);
  return map.find(s)->second;
}

ComplexValue &VariableValueMap::take(const ScopedIdentifier &s) {
  auto it = map.find(s);
  ComplexValue &cv = it->second;
  map.erase(it);
  auto changed_it = changed.find(s);
  if (changed_it!=changed.end()) {
    changed.erase(changed_it);
  }
  return cv;
}
void VariableValueMap::update(const ScopedIdentifier &s, ComplexValue &cv) {
  map.find(s)->second = cv;
  changed.insert(s);
}
bool VariableValueMap::has(const ScopedIdentifier &s) {
  return map.find(s)!=map.end();
}
void VariableValueMap::resetChangeFlags() {
  changed.clear();
}
std::unordered_set<ScopedIdentifier> VariableValueMap::changedEntries() const {
  return changed;
}


////////////////////////////////////////////
////          SpecialVectorizer         ////
////////////////////////////////////////////

void SpecialVectorizer::visit(Block &elem) {
  ScopedVisitor::enterScope(elem);
  variableValues.resetChangeFlags();

  for (auto &p: elem.getStatementPointers()) {
    p->accept(*this);
    if (delete_flag) { p.reset(); }
    delete_flag = false;
  }
  elem.removeNullStatements();

  // TODO: Emit all relevant assignments again!
  for (auto &scopedID: variableValues.changedEntries()) {
    auto &cv = variableValues.take(scopedID);
    for (auto &statement : cv.statementsToExecutePlan()) {
      elem.appendStatement(std::move(statement));
    }
  }
  variableValues.resetChangeFlags();
  ScopedVisitor::exitScope();
}

void SpecialVectorizer::visit(Assignment &elem) {

  /// current scope
  auto &scope = getCurrentScope();

  /// target of the assignment
  AbstractTarget &target = elem.getTarget();
  ScopedIdentifier targetID;
  BatchingConstraint targetBatchingConstraint;

  // We currently assume that the target has either the form <Variable> or <Variable>[<LiteralInt>]
  if (target.countChildren()==0) {
    auto variable = dynamic_cast<Variable &>(target);
    auto id = variable.getIdentifier();
    targetID = scope.resolveIdentifier(id);
    if (constraints.find(targetID)!=constraints.end()) {
      auto t = constraints.find(targetID)->second.getSlot();
      targetBatchingConstraint = BatchingConstraint(t, targetID);
    }
  } else {
    auto indexAccess = dynamic_cast<IndexAccess &>(target);
    auto variable = dynamic_cast<Variable &>(indexAccess.getTarget());
    auto index = dynamic_cast<LiteralInt &>(indexAccess.getIndex());
    targetID = scope.resolveIdentifier(variable.getIdentifier());
    targetBatchingConstraint = BatchingConstraint(index.getValue(), targetID);
  }

  /// Optimize the value of the assignment
  auto cv = batchExpression(elem.getValue(), targetBatchingConstraint);

  /// Combine the execution plans, if they already exist
  if (variableValues.has(targetID)) {
    variableValues.getToModify(targetID).merge(cv);
  } else {
    precomputedValues.push_back(cv);
    variableValues.add(targetID, precomputedValues[precomputedValues.size() - 1]);
  }

  // Now delete this assignment
  delete_flag = true;
}

std::string SpecialVectorizer::getAuxiliaryInformation() {
  //TODO: Implement returning of auxiliary information
  return "NOT IMPLEMENTED YET";
}

ComplexValue SpecialVectorizer::batchExpression(AbstractExpression &exp, BatchingConstraint) {
  //TODO: IMPLEMENT
  return ComplexValue(exp);
}