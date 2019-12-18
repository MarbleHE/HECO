#include <utility>


#include "../../include/ast/Variable.h"

Variable::Variable(std::string identifier) : identifier(std::move(identifier)) {}

json Variable::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["identifier"] = this->identifier;
    return j;
}

void Variable::accept(Visitor &v) {
    v.visit(*this);
}

std::string Variable::getNodeName() const {
    return "Variable";
}

const std::string &Variable::getIdentifier() const {
    return identifier;
}
