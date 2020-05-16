#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/Variable.h"

MatrixElementRef::MatrixElementRef(AbstractExpr *mustEvaluateToAbstractLiteral,
                                   int rowIndex,
                                   int columnIndex) {
  setAttributes(mustEvaluateToAbstractLiteral, new LiteralInt(rowIndex), new LiteralInt(columnIndex));
}

MatrixElementRef::MatrixElementRef(AbstractExpr *mustEvaluateToAbstractLiteral,
                                   AbstractExpr *rowIndex,
                                   AbstractExpr *columnIndex) {
  setAttributes(mustEvaluateToAbstractLiteral, rowIndex, columnIndex);
}

void MatrixElementRef::setAttributes(AbstractExpr *elementContainingMatrix,
                                     AbstractExpr *rowIndex,
                                     AbstractExpr *columnIndex) {
  removeChildren();
  addChildren({elementContainingMatrix, rowIndex, columnIndex}, true);
}

MatrixElementRef::MatrixElementRef(AbstractExpr *mustEvaluateToAbstractLiteral, AbstractExpr *rowIndex) {
  setAttributes(mustEvaluateToAbstractLiteral, rowIndex, nullptr);
}

MatrixElementRef::MatrixElementRef(AbstractExpr *mustEvaluateToAbstractLiteral, int rowIndex) {
  setAttributes(mustEvaluateToAbstractLiteral, new LiteralInt(rowIndex), nullptr);
}

std::string MatrixElementRef::getNodeType() const {
  return std::string("MatrixElementRef");
}

void MatrixElementRef::accept(Visitor &v) {
  v.visit(*this);
}

AbstractNode *MatrixElementRef::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new MatrixElementRef(
      getOperand()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
      getRowIndex()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
      getColumnIndex()==nullptr ? nullptr :
      getColumnIndex()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

json MatrixElementRef::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["operand"] = getOperand()->toJson();
  j["rowIndex"] = getRowIndex()->toJson();
  j["columnIndex"] = getColumnIndex()->toJson();
  return j;
}

AbstractExpr *MatrixElementRef::getOperand() const {
  return dynamic_cast<AbstractExpr *>(getChildAtIndex(0));
}

AbstractExpr *MatrixElementRef::getRowIndex() const {
  return dynamic_cast<AbstractExpr *>(getChildAtIndex(1));
}

AbstractExpr *MatrixElementRef::getColumnIndex() const {
  return dynamic_cast<AbstractExpr *>(getChildAtIndex(2));
}

std::vector<std::string> MatrixElementRef::getVariableIdentifiers() {
  std::vector<std::string> resultVec;
  auto insertIfNonEmpty = [&resultVec](std::vector<std::string> vec) {
    if (!vec.empty()) resultVec.insert(resultVec.end(), vec.begin(), vec.end());
  };
  insertIfNonEmpty(getOperand()->getVariableIdentifiers());
  insertIfNonEmpty(getRowIndex()->getVariableIdentifiers());
  insertIfNonEmpty(getColumnIndex()->getVariableIdentifiers());
  return resultVec;
}

std::vector<Variable *> MatrixElementRef::getVariables() {
  std::vector<Variable*> resultVec;
  auto insertIfNonEmpty = [&resultVec](std::vector<Variable*> vec) {
    if (!vec.empty()) resultVec.insert(resultVec.end(), vec.begin(), vec.end());
  };
  insertIfNonEmpty(getOperand()->getVariables());
  insertIfNonEmpty(getRowIndex()->getVariables());
  insertIfNonEmpty(getColumnIndex()->getVariables());
  return resultVec;
}

bool MatrixElementRef::contains(Variable *var) {
  auto vars = getVariableIdentifiers();
  return std::count(vars.begin(), vars.end(), var->getIdentifier()) > 0;
}

bool MatrixElementRef::isEqual(AbstractExpr *other) {
  if (auto otherAsGME = dynamic_cast<MatrixElementRef *>(other)) {
    return getOperand()->isEqual(otherAsGME->getOperand())
        && getRowIndex()->isEqual(otherAsGME->getRowIndex())
        && getColumnIndex()->isEqual(otherAsGME->getColumnIndex());
  }
  return false;
}

int MatrixElementRef::getMaxNumberChildren() {
  return 3;
}

std::string MatrixElementRef::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

AbstractNode *MatrixElementRef::cloneFlat() {
  return new MatrixElementRef(nullptr, nullptr, nullptr);
}

bool MatrixElementRef::supportsCircuitMode() {
  return true;
}
