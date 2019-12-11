#include "../../include/ast/Call.h"

Call::Call(AbstractExpr *callee, const std::vector<FunctionParameter> &arguments) : callee(callee),
                                                                                    arguments(arguments) {}


json Call::toJson() const {
    json j;
    j["type"] = "Call";
    j["arguments"] = this->arguments;
    j["callee"] = this->callee->toString();
    return j;
}
