#include <utility>
#include "VarAssignm.h"
#include "BinaryExpr.h"

json VarAssignm::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["identifier"] = this->identifier;
    j["value"] = this->value->toJson();
    return j;
}

VarAssignm::VarAssignm(std::string identifier, AbstractExpr *value) : identifier(std::move(identifier)), value(value) {}

void VarAssignm::accept(Visitor &v) {
    v.visit(*this);
}

const std::string &VarAssignm::getIdentifier() const {
    return identifier;
}

AbstractExpr *VarAssignm::getValue() const {
    return value;
}

std::string VarAssignm::getNodeName() const {
    return "VarAssignm";
}

BinaryExpr *VarAssignm::contains(BinaryExpr *bexpTemplate, BinaryExpr *excludedSubtree) {
    return this->getValue()->contains(bexpTemplate, excludedSubtree);
}

VarAssignm::~VarAssignm() {
    delete value;
}

std::string VarAssignm::getVarTargetIdentifier() {
    return this->getIdentifier();
}

bool VarAssignm::isEqual(AbstractStatement *as) {
    if (auto otherVarAssignm = dynamic_cast<VarAssignm *>(as)) {
        return this->getIdentifier() == otherVarAssignm->getIdentifier() &&
                this->getValue()->isEqual(otherVarAssignm->getValue());
    }
    return false;
}
Literal* VarAssignm::evaluate(Ast &ast) {
  ast.updateVarValue(this->getIdentifier(), this->getValue()->evaluate(ast));
  return nullptr;
}
