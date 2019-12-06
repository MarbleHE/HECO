#include "../../include/ast/VarDecl.h"


VarDecl::VarDecl(const std::string &name, const std::string &datatype) : name(name), datatype(datatype) {}

VarDecl::VarDecl(const std::string &name, const std::string &datatype, AbstractExpr *initializer) : name(name),
                                                                                                    datatype(datatype),
                                                                                                    initializer(
                                                                                                            initializer) {}

void VarDecl::print() {
    printf("VarDecl { name: %s, datatype: %s, initializer: ", name.c_str(), datatype.c_str());
    initializer->print();
    printf("}\n");

}


