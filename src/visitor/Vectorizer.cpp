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
BatchingConstraint::BatchingConstraint(int slot, const std::string &identifier) : slot(slot), identifier(identifier) {}

int BatchingConstraint::getSlot() const {
  return slot;
}
void BatchingConstraint::setSlot(int slot_) {
  slot = slot_;
}
const std::string &BatchingConstraint::getIdentifier() const {
  return identifier;
}
void BatchingConstraint::setIdentifier(const std::string &identifier_) {
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

////////////////////////////////////////////
////          SpecialVectorizer         ////
////////////////////////////////////////////

void SpecialVectorizer::visit(Block &elem) {
  ScopedVisitor::enterScope(elem);
  for (auto &p: elem.getStatementPointers()) {
    p->accept(*this);
    if (delete_flag) { p.reset(); }
    delete_flag = false;
  }
  // TODO: Emit all relevant assignments again!
  ScopedVisitor::exitScope();
}

void SpecialVectorizer::visit(Assignment &elem) {

  /// current scope
  auto &scope = getCurrentScope();

  /// target of the assignment
  AbstractTarget &target = elem.getTarget();
  ScopedIdentifier targetID(scope, ""); // Dummy, since no default ctor
  BatchingConstraint batchingConstraint(-1, "");

  // We currently assume that the target has either the form <Variable> or <Variable>[<LiteralInt>]
  if (target.countChildren()==0) {
    auto variable = dynamic_cast<Variable &>(target);
    auto id = variable.getIdentifier();
    targetID = scope.resolveIdentifier(id);
    if (constraints.find(targetID)!=constraints.end()) {
      auto t = constraints.find(targetID)->second.getSlot();
      batchingConstraint.setSlot(t);
    }
  } else {
    auto indexAccess = dynamic_cast<IndexAccess &>(target);
    auto variable = dynamic_cast<Variable &>(indexAccess.getTarget());
    auto index = dynamic_cast<LiteralInt &>(indexAccess.getIndex());
    targetID = scope.resolveIdentifier(variable.getIdentifier());
    batchingConstraint.setSlot(index.getValue());
  }

  /// Optimize the value of the assignment
  auto cv = batchExpression(elem.getValue(), batchingConstraint);

  /// Combine the execution plans, if they already exist
  auto it = variableValues.find(targetID);
  if (it!=variableValues.end()) {
    it->second.merge(cv);
  } else {
    precomputedValues.push_back(cv);
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
