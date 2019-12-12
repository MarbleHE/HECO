
#include "../../include/ast/Variable.h"

Variable::Variable(const std::string &identifier) : identifier(identifier) {}

json Variable::toJson() const {
    json j;
    j["type"] = "Variable";
    j["identifier"] = this->identifier;
    return j;
}

void Variable::accept(Visitor &v) {
    v.visit(*this);
}
