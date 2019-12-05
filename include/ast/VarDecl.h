#ifndef MASTER_THESIS_CODE_VARDECL_H
#define MASTER_THESIS_CODE_VARDECL_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarDecl : public AbstractStatement {
public:
    std::string name;
    std::string datatype;
    AbstractExpr* initializer;
    VarDecl(const std::string &name, const std::string &datatype);
    VarDecl(const std::string &name, const std::string &datatype, AbstractExpr *initializer);
};


#endif //MASTER_THESIS_CODE_VARDECL_H
