#include "../../include/ast/Class.h"

#include <utility>

Class::Class(const std::string &name, const std::string &superclass, const std::vector<Function> &methods) : name(name),
                                                                                                             superclass(
                                                                                                                     superclass),
                                                                                                             methods(methods) {}

json Class::toJson() const {
    json j;
    j["type"] = "Class";
    j["name"] = this->name;
    j["superclass"] = this->superclass;
    j["methods"] = this->methods;
    return j;
}