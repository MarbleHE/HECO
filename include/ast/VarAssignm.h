#ifndef MASTER_THESIS_CODE_VARASSIGNM_H
#define MASTER_THESIS_CODE_VARASSIGNM_H


#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarAssignm : public AbstractStatement, public Node {
private:
    std::string identifier;
    AbstractExpr *value;
public:
    VarAssignm(const std::string &identifier, AbstractExpr *value);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    const std::string &getIdentifier() const;

    AbstractExpr *getValue() const;

    std::string getNodeName() const override;

    BinaryExpr *contains(BinaryExpr *bexpTemplate) override;

    virtual ~VarAssignm();
};

#endif //MASTER_THESIS_CODE_VARASSIGNM_H
