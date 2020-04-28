#include "ast_opt/ast/GetMatrixSize.h"
#include "ast_opt/ast/AbstractExpr.h"
#include "ast_opt/ast/LiteralInt.h"

json GetMatrixSize::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["matrix"] = getMatrixOperand()->toJson();
  j["dimension"] = getDimensionParameter()->toJson();
  return j;
}

bool GetMatrixSize::isEqual(AbstractExpr *other) {
  if (auto otherGetMatrixSize = dynamic_cast<GetMatrixSize *>(other)) {
    return getMatrixOperand()->isEqual(otherGetMatrixSize->getMatrixOperand())
        && getDimensionParameter()->isEqual(otherGetMatrixSize->getDimensionParameter());
  }
  return false;
}

std::string GetMatrixSize::getNodeType() const {
  return std::string("GetMatrixSize");
}

int GetMatrixSize::getMaxNumberChildren() {
  return 2;
}

void GetMatrixSize::accept(Visitor &v) {
  v.visit(*this);
}

std::string GetMatrixSize::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

AbstractNode *GetMatrixSize::clone(bool keepOriginalUniqueNodeId) {
  return new GetMatrixSize(getMatrixOperand()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                           getDimensionParameter()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
}

bool GetMatrixSize::supportsCircuitMode() {
  return true;
}

GetMatrixSize::GetMatrixSize(AbstractExpr *matrix, AbstractExpr *requestedDimension) {
  removeChildren();
  addChildren({matrix, requestedDimension}, true);
}

AbstractExpr *GetMatrixSize::getMatrixOperand() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(0));
}

AbstractExpr *GetMatrixSize::getDimensionParameter() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(1));
}

GetMatrixSize::~GetMatrixSize() = default;
