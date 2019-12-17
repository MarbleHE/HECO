#include "../../include/ast/VarDecl.h"

#include <utility>
#include <LiteralInt.h>
#include "BinaryExpr.h"

json VarDecl::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["identifier"] = identifier;
    j["datatype"] = datatype;
//    if (this->initializer != nullptr) {
//        j["initializer"] = this->initializer->toJson();
//    } else {
    j["initializer"] = ""; // FIXME
//    }
    return j;
}

VarDecl::VarDecl(std::string name, std::string datatype, AbstractExpr *initializer)
        : identifier(name), datatype(datatype), initializer(std::move(initializer)) {

}

VarDecl::VarDecl(std::string name, std::string datatype) : identifier(name), datatype(datatype) {
    this->initializer = nullptr;
}

VarDecl::VarDecl(std::string name, std::string datatype, int i) : identifier(name), datatype(datatype) {
    if (datatype != "int") {
        throw std::logic_error("test");
    } else {
        initializer = new LiteralInt(i);
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

BinaryExpr *VarDecl::contains(BinaryExpr *bexpTemplate) {
    if (auto *castedBexp = dynamic_cast<BinaryExpr *>(this->getInitializer())) {
        return castedBexp->containsValuesFrom(bexpTemplate);
    }
    return nullptr;
}

VarDecl::~VarDecl() {
    delete initializer;
}
