#include "../../include/ast/VarDecl.h"

#include <utility>


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

VarDecl::VarDecl(std::string name, std::string datatype, std::unique_ptr<AbstractExpr> initializer)
        : name(name), datatype(datatype), initializer(std::move(initializer)) {

}

VarDecl::VarDecl(std::string name, std::string datatype) : name(name), datatype(datatype) {
}
