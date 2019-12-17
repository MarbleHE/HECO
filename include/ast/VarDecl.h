#ifndef MASTER_THESIS_CODE_VARDECL_H
#define MASTER_THESIS_CODE_VARDECL_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarDecl : public AbstractStatement, public Node {
private:
    std::string identifier;
    std::string datatype;
    AbstractExpr *initializer;

public:

    VarDecl(std::string name, std::string datatype);

    VarDecl(std::string name, std::string datatype, AbstractExpr *initializer);

    VarDecl(std::string name, std::string datatype, int i);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    std::string getNodeName() const override;

    const std::string &getIdentifier() const;

    const std::string &getDatatype() const;

    AbstractExpr *getInitializer() const;

    BinaryExpr *contains(BinaryExpr *bexpTemplate) override;

    virtual ~VarDecl();
};


#endif //MASTER_THESIS_CODE_VARDECL_H
