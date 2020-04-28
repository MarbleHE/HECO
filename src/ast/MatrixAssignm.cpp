#include "ast_opt/ast/MatrixAssignm.h"
#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/ArithmeticExpr.h"

MatrixAssignm::~MatrixAssignm() = default;

MatrixAssignm::MatrixAssignm(MatrixElementRef *assignmentTarget, AbstractExpr *value) {
  setAttributes(reinterpret_cast<AbstractExpr *>(assignmentTarget), value);
}

void MatrixAssignm::setAttributes(AbstractExpr *assignmTarget, AbstractExpr *value) {
  removeChildren();
  addChildren({assignmTarget, value}, true);
}

std::string MatrixAssignm::getNodeType() const {
  return std::string("MatrixAssignm");
}

int MatrixAssignm::getMaxNumberChildren() {
  return 2;
}

void MatrixAssignm::accept(Visitor &v) {
  v.visit(*this);
}

std::string MatrixAssignm::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

MatrixElementRef *MatrixAssignm::getAssignmTarget() const {
  return reinterpret_cast<MatrixElementRef *>(getChildAtIndex(0, true));
}

std::string MatrixAssignm::getAssignmTargetString() const {
  std::stringstream ss;
  // Build a string in the format <matrixIdentifier>[<rowIndexExpr>][<columnIndexExpr>].
  // Unfortunately, toString also includes a node's name such that we get a string like
  // "Variable (M)[LiteralInt (32)][LiteralInt (1)]" where the compacter "M[32][1]" would be desired instead.
  ss << getAssignmTarget()->getOperand()->toString(true);
  ss << "[";
  ss << getAssignmTarget()->getRowIndex()->toString(true);
  ss << "]" << "[";
  ss << getAssignmTarget()->getColumnIndex()->toString(true);
  ss << "]";
  std::string res(ss.str());
  // remove any new line characters
  res.erase(std::remove_if(res.begin(), res.end(), [](char c) { return c=='\n'; }), res.end());
  // replace tabs by whitespace characters
  std::replace(res.begin(), res.end(), '\t', ' ');
  return res;
}

AbstractExpr *MatrixAssignm::getValue() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(1, true));
}

AbstractNode *MatrixAssignm::clone(bool keepOriginalUniqueNodeId) {
  return new MatrixAssignm(getAssignmTarget()->clone(keepOriginalUniqueNodeId)->castTo<MatrixElementRef>(),
                           getValue()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
}

bool MatrixAssignm::supportsCircuitMode() {
  return true;
}

json MatrixAssignm::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["target"] = getAssignmTarget()->toJson();
  j["value"] = getValue() ? getValue()->toJson() : "";
  return j;
}

AbstractBinaryExpr *MatrixAssignm::contains(AbstractBinaryExpr *aexpTemplate, ArithmeticExpr *excludedSubtree) {
  return getValue()->contains(aexpTemplate, excludedSubtree);
}

bool MatrixAssignm::isEqual(AbstractStatement *as) {
  if (auto otherMxAssignm = dynamic_cast<MatrixAssignm *>(as)) {
    return getAssignmTarget()->isEqual(otherMxAssignm->getAssignmTarget())
        && getValue()->isEqual(otherMxAssignm->getValue());
  }
  return false;
}
