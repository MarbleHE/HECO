#ifndef MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
#define MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H

#include <string>
#include <nlohmann/json.hpp>
#include "../visitor/Visitor.h"
#include "Node.h"

using json = nlohmann::json;

class AbstractStatement : public Node {
public:
    virtual ~AbstractStatement() = default;

    [[nodiscard]] std::string toString() const override;

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    virtual BinaryExpr *contains(BinaryExpr *bexpTemplate, BinaryExpr *excludedSubtree);

    virtual std::string getVarTargetIdentifier();

    virtual bool isEqual(AbstractStatement *as);

    Literal *evaluate(Ast &ast) override;
};

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj);

/// JSON representation to be used for vector<AbstractStatement> objects.
void to_json(json &j, const AbstractStatement &absStat);

void to_json(json &j, const AbstractStatement *absStat);

#endif //MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
