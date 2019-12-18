#include "../../include/ast/Class.h"


Class::Class(std::string name, std::string superclass, std::vector<Function> methods)
        : name(std::move(name)), superclass(std::move(superclass)), methods(std::move(methods)) {}

json Class::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["identifier"] = this->name;
    j["superclass"] = this->superclass;
    j["methods"] = this->methods;
    return j;
}

void Class::accept(Visitor &v) {
    v.visit(*this);
}

const std::string &Class::getName() const {
    return name;
}

const std::string &Class::getSuperclass() const {
    return superclass;
}

const std::vector<Function> &Class::getMethods() const {
    return methods;
}

std::string Class::getNodeName() const {
    return "Class";
}
