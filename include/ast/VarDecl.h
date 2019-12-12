#ifndef MASTER_THESIS_CODE_VARDECL_H
#define MASTER_THESIS_CODE_VARDECL_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarDecl : public AbstractStatement {
public:
    std::string name;
    std::string datatype;
    AbstractExpr *initializer;

    VarDecl(std::string name, std::string datatype);

    VarDecl(std::string name, std::string datatype, AbstractExpr *initializer);

    VarDecl(std::string name, std::string datatype, int i);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

};


#endif //MASTER_THESIS_CODE_VARDECL_H
