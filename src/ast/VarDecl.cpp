#include "../../include/ast/VarDecl.h"

#include <utility>
#include <LiteralInt.h>
#include "BinaryExpr.h"
#include "Group.h"

json VarDecl::toJson() const {
    json j = {{"type",       getNodeName()},
              {"identifier", identifier},
              {"datatype",   datatype}};
    if (this->initializer != nullptr) {
        j["initializer"] = this->initializer->toJson();
    }
    return j;
}

VarDecl::VarDecl(std::string name, std::string datatype, AbstractExpr *initializer)
        : identifier(std::move(std::move(name))), datatype(std::move(std::move(datatype))), initializer(initializer) {

}

VarDecl::VarDecl(std::string name, std::string datatype) : identifier(std::move(std::move(name))),
                                                           datatype(std::move(std::move(datatype))) {
    this->initializer = nullptr;
}

VarDecl::VarDecl(std::string name, const std::string &datatype, int i) : identifier(std::move(std::move(name))),

                                                                         datatype(datatype) {
    if (datatype == "int") {
        initializer = new LiteralInt(i);
    } else {
        throw std::logic_error("test");
    }
}

void VarDecl::accept(Visitor &v) {
    v.visit(*this);
}

std::string VarDecl::getNodeName() const {
    return "VarDecl";
}

const std::string &VarDecl::getIdentifier() const {
    return identifier;
}

const std::string &VarDecl::getDatatype() const {
    return datatype;
}

AbstractExpr *VarDecl::getInitializer() const {
    return initializer;
}

BinaryExpr *VarDecl::contains(BinaryExpr *bexpTemplate, BinaryExpr *excludedSubtree) {
    return this->getInitializer()->contains(bexpTemplate, excludedSubtree);
}

VarDecl::~VarDecl() {
    delete initializer;
}

std::string VarDecl::getVarTargetIdentifier() {
    return this->getIdentifier();
}
