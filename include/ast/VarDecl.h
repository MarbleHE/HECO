#ifndef MASTER_THESIS_CODE_VARDECL_H
#define MASTER_THESIS_CODE_VARDECL_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarDecl : public AbstractStatement {
public:
    std::string name;
    std::string datatype;
    std::unique_ptr<AbstractExpr> initializer;

    VarDecl(std::string name, std::string datatype);

    VarDecl(std::string name, std::string datatype, std::unique_ptr<AbstractExpr> initializer);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_VARDECL_H
