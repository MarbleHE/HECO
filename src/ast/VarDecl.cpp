#include "../../include/ast/VarDecl.h"


VarDecl::VarDecl(const std::string &name, const std::string &datatype) : name(name), datatype(datatype) {}

VarDecl::VarDecl(const std::string &name, const std::string &datatype, AbstractExpr *initializer) : name(name),
                                                                                                    datatype(datatype),
                                                                                                    initializer(
                                                                                                            initializer) {}


