#ifndef MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
#define MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H

#include <string>
#include <nlohmann/json.hpp>
#include "../visitor/Visitor.h"
#include "Node.h"

using json = nlohmann::json;


class AbstractStatement {
public:
    virtual ~AbstractStatement() = default;

    virtual std::string toString() const;

    virtual json toJson() const;

    virtual void accept(Visitor &v);

    virtual BinaryExpr *contains(BinaryExpr *bexpTemplate);

};

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj);


/// JSON representation to be used for vector<AbstractStatement> objects.
void to_json(json &j, const AbstractStatement &absStat);


#endif //MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
