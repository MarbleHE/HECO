
#include "../../include/ast/VarAssignm.h"

VarAssignm::VarAssignm(const std::string &identifier, AbstractExpr *value) : identifier(identifier), value(value) {}
