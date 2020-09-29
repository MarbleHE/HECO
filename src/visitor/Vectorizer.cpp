#include "ast_opt/ast/AbstractTarget.h"
#include "ast_opt/ast/Assignment.h"
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
  return getSlot() != -1;
}

////////////////////////////////////////////
////           ComplexValue             ////
////////////////////////////////////////////
ComplexValue::ComplexValue(AbstractExpression &) {
  //TODO: Implement ComplexValue Ctor
}

BatchingConstraint& ComplexValue::getBatchingConstraint() {
  //TODO: Implement ComplexValue::getBatchingConstraint
  return batchingConstraint;
}
////////////////////////////////////////////
////          SpecialVectorizer         ////
////////////////////////////////////////////
SpecialVectorizer::SpecialVectorizer(TypeMap types, VariableValueMap values, ConstraintMap constraints, RotationMap rotations) :
    types(types), variableValues(values), constraints(constraints), rotations(rotations){}

bool SpecialVectorizer::isInExpressionValues(const AbstractNode &elem) const {
  return expressionValues.find(valueHash(elem)) != expressionValues.end();
}

std::string SpecialVectorizer::valueHash(const AbstractNode &elem) const {
  auto it = valueHashMap.find(elem.getUniqueNodeId());
  if(it != valueHashMap.end()) {
    return it->second;
  } else {
    throw std::runtime_error("No pre-created Hash found for " + elem.getUniqueNodeId());
  }
}

void SpecialVectorizer::visit(Assignment &elem) {

  /// current scope
  auto &scope = getCurrentScope();

  /// target of the assignment
  AbstractTarget& target = elem.getTarget();
  ScopedIdentifier targetID(scope, ""); // Dummy, since no default ctor
  BatchingConstraint batchingConstraint(-1, "");

  // TODO: We currently assume that the target has either the form <Variable> or <Variable>[<LiteralInt>]
  if (target.countChildren() == 0) {
    auto variable = dynamic_cast<Variable&>(target);
    auto id = variable.getIdentifier();
    targetID = scope.resolveIdentifier(id);
    if(constraints.find(targetID) != constraints.end()) {
      auto t = constraints.find(targetID)->second.getSlot();
      batchingConstraint.setSlot(t);
    }
  } else {
    auto indexAccess = dynamic_cast<IndexAccess&>(target);
    auto variable = dynamic_cast<Variable&>(indexAccess.getTarget());
    auto index = dynamic_cast<LiteralInt&>(indexAccess.getIndex());
    targetID = scope.resolveIdentifier(variable.getIdentifier());
    batchingConstraint.setSlot(index.getValue());
  }

  // Push the target slot to the target-slot-stack
  targetSlotStack.push(batchingConstraint.getSlot());

  /// Value of the assignment
  AbstractExpression& value = elem.getValue();

  // TODO: Call hash-creating visitor

  if(isInExpressionValues(value)) {
    // We have this expression pre-computed
    ComplexValue &cv = expressionValues.find(valueHash(value))->second;

    // Check how to get it to the target slot
    if(!batchingConstraint.hasTargetSlot()) {
      // We have no fixed target slot
      // Simply register the existing value in the variableValueMap
      variableValues.insert({targetID,cv});
    }
  } else {
    // Build an execution plan (complex value) from the expression via recursion
    value.accept(*this);

    // Retrieve that execution plan from the stack
    ComplexValue cv = resultValueStack.top();
    resultValueStack.pop();

    // Insert it into expressionValues:
    expressionValues.insert({valueHash(value),cv});

    // Register the value in variableValues (must be the object in expressionValues since local cv will be destroyed)
    variableValues.insert({targetID,expressionValues.find(valueHash(value))->second});
  }

  // Remove the current target slot
  targetSlotStack.pop();
}

std::string SpecialVectorizer::getAuxiliaryInformation() {
  //TODO: Implement returning of auxiliary information
  return "NOT IMPLEMENTED YET";
}