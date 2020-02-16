#include "Return.h"

#include <utility>

json Return::toJson() const {
  json j;
  j["type"] = getNodeName();

  // build the string of result values
  json array = json::array();
  for (auto &expr : getReturnExpressions()) array.push_back(expr->toJson());
  j["values"] = array;

  return j;
}

Return::Return(AbstractExpr* returnValue) {
  setAttributes({returnValue});
}

Return::Return(std::vector<AbstractExpr*> returnValues) {
  setAttributes(std::move(returnValues));
}

void Return::accept(Visitor &v) {
  v.visit(*this);
}

std::string Return::getNodeName() const {
  return "Return";
}

Return::~Return() {
  for (auto &child : getChildren()) delete child;
}

std::vector<Literal*> Return::evaluate(Ast &ast) {
  std::vector<Literal*> result;
  for (auto &expr : getReturnExpressions()) {
    auto exprEvaluationResult = expr->evaluate(ast);
    result.insert(result.end(), exprEvaluationResult.begin(), exprEvaluationResult.end());
  }
  return result;
}

Return::Return() = default;

int Return::getMaxNumberChildren() {
  return -1;
}

void Return::setAttributes(std::vector<AbstractExpr*> returnExpr) {
  removeChildren();
  // we need to convert vector of AbstractExpr* into vector of Node* prior calling addChildren
  std::vector<Node*> returnExprAsNodes(returnExpr.begin(), returnExpr.end());
  addChildren(returnExprAsNodes, false);
  addParentTo(this, returnExprAsNodes);
}

std::vector<AbstractExpr*> Return::getReturnExpressions() const {
  std::vector<AbstractExpr*> vec;
  for (auto &child : getChildrenNonNull()) vec.push_back(child->castTo<AbstractExpr>());
  return vec;
}

bool Return::supportsCircuitMode() {
  return true;
}

Node* Return::createClonedNode(bool keepOriginalUniqueNodeId) {
  std::vector<AbstractExpr*> returnValues;
  for (auto &child : getReturnExpressions())
    returnValues.push_back(child->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  return new Return(returnValues);
}

