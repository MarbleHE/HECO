#include "../../include/ast/VarDecl.h"

#include <utility>
#include <LiteralInt.h>


json VarDecl::toJson() const {
    json j;
    j["type"] = "VarDecl";
    j["name"] = name;
    j["datatype"] = datatype;
    if (this->initializer != nullptr) {
        j["initializer"] = this->initializer->toJson();
    } else {
        j["initializer"] = "";
    }
    return j;
}

VarDecl::VarDecl(std::string name, std::string datatype, AbstractExpr *initializer)
        : name(name), datatype(datatype), initializer(std::move(initializer)) {

}

VarDecl::VarDecl(std::string name, std::string datatype) : name(name), datatype(datatype) {
}

VarDecl::VarDecl(std::string name, std::string datatype, int i) : name(name), datatype(datatype) {
    if (datatype != "int") {
        throw std::logic_error("test");
    } else {
        initializer = new LiteralInt(i);
    }
}
