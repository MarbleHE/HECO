#include <iostream>
#include "../../include/ast/AbstractStatement.h"

std::string AbstractStatement::toString() const {
    return this->toJson().dump();
}

json AbstractStatement::toJson() const {
    return json({"type", "AbstractStatement"});
}

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj) {
    return outs << obj.toString();
}

void to_json(json &j, const std::unique_ptr<AbstractStatement> &param) {
    j = param->toJson();
}
