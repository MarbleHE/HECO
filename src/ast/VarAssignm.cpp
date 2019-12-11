
#include "../../include/ast/VarAssignm.h"

json VarAssignm::toJson() const {
    json j;
    j["type"] = "VarAssignm";
    j["identifier"] = this->identifier;
    j["value"] = this->value->toJson();
    return j;
}

VarAssignm::VarAssignm(const std::string &identifier, std::unique_ptr<AbstractExpr> value) : identifier(identifier),
                                                                                             value(std::move(value)) {

}
