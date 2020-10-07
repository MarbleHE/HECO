#include "ast_opt/utilities/ComplexValue.h"

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