#ifndef MASTER_THESIS_CODE_VARIABLE_H
#define MASTER_THESIS_CODE_VARIABLE_H


#include <string>
#include "AbstractExpr.h"

class Variable : public AbstractExpr, public Node {
private:
    std::string identifier;

public:
    explicit Variable(std::string identifier);

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    [[nodiscard]] const std::string &getIdentifier() const;
};


#endif //MASTER_THESIS_CODE_VARIABLE_H
