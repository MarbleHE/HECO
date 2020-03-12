#include "GetMatrixElement.h"
#include "Variable.h"

GetMatrixElement::GetMatrixElement(AbstractExpr *mustEvaluateToAbstractLiteral,
                                   int rowIndex,
                                   int columnIndex) {
  setAttributes(mustEvaluateToAbstractLiteral, new LiteralInt(rowIndex), new LiteralInt(columnIndex));
}

GetMatrixElement::GetMatrixElement(AbstractExpr *mustEvaluateToAbstractLiteral,
                                   AbstractExpr *rowIndex,
                                   AbstractExpr *columnIndex) {
  setAttributes(mustEvaluateToAbstractLiteral, rowIndex, columnIndex);
}

void GetMatrixElement::setAttributes(AbstractExpr *elementContainingMatrix,
                                     AbstractExpr *rowIndex,
                                     AbstractExpr *columnIndex) {
  removeChildren();
  addChildren({elementContainingMatrix, rowIndex, columnIndex}, true);
}

std::string GetMatrixElement::getNodeType() const {
  return std::string("GetMatrixElement");
}

void GetMatrixElement::accept(Visitor &v) {
  v.visit(*this);
}

AbstractNode *GetMatrixElement::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new GetMatrixElement(
      getOperand()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
      getRowIndex()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
      getColumnIndex()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

json GetMatrixElement::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["operand"] = getOperand()->toJson();
  j["rowIndex"] = getRowIndex()->toJson();
  j["columnIndex"] = getColumnIndex()->toJson();
  return j;
}

AbstractExpr *GetMatrixElement::getOperand() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(0));
}

AbstractExpr *GetMatrixElement::getRowIndex() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(1));
}

AbstractExpr *GetMatrixElement::getColumnIndex() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(2));
}

std::vector<std::string> GetMatrixElement::getVariableIdentifiers() {
  std::vector<std::string> resultVec;
  auto insertIfNonEmpty = [&resultVec](std::vector<std::string> vec) {
    if (!vec.empty()) resultVec.insert(resultVec.end(), vec.begin(), vec.end());
  };
  insertIfNonEmpty(getOperand()->getVariableIdentifiers());
  insertIfNonEmpty(getRowIndex()->getVariableIdentifiers());
  insertIfNonEmpty(getColumnIndex()->getVariableIdentifiers());
  return resultVec;
}

bool GetMatrixElement::contains(Variable *var) {
  auto vars = getVariableIdentifiers();
  return std::count(vars.begin(), vars.end(), var->getIdentifier()) > 0;
}

bool GetMatrixElement::isEqual(AbstractExpr *other) {
  if (auto otherAsGME = dynamic_cast<GetMatrixElement *>(other)) {
    return getOperand()->isEqual(otherAsGME->getOperand())
        && getRowIndex()->isEqual(otherAsGME->getRowIndex())
        && getColumnIndex()->isEqual(otherAsGME->getColumnIndex());
  }
  return false;
}

int GetMatrixElement::getMaxNumberChildren() {
  return 3;
}

std::string GetMatrixElement::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

AbstractNode *GetMatrixElement::cloneFlat() {
  return new GetMatrixElement(nullptr, nullptr, nullptr);
}

bool GetMatrixElement::supportsCircuitMode() {
  return true;
}
