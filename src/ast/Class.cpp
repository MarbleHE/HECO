#include "../../include/ast/Class.h"

#include <utility>

Class::Class(std::string name, Class *superclass, std::vector<Function> methods) : name(std::move(name)),
                                                                                                 superclass(superclass),
                                                                                                 methods(std::move(methods)) {}