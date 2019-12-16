#ifndef MASTER_THESIS_CODE_VARIABLE_H
#define MASTER_THESIS_CODE_VARIABLE_H


#include <string>
#include "AbstractExpr.h"

class Variable : public AbstractExpr, public Node {
private:
    std::string identifier;

public:
    Variable(const std::string &identifier);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    std::string getNodeName() const override;

    const std::string &getIdentifier() const;
};


#endif //MASTER_THESIS_CODE_VARIABLE_H
