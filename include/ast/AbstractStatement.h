#ifndef MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
#define MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H

#include <string>
#include <nlohmann/json.hpp>
#include "../visitor/Visitor.h"

using json = nlohmann::json;


class AbstractStatement {
public:
    virtual ~AbstractStatement() = default;

    virtual std::string toString() const;

    virtual json toJson() const;

    virtual void accept(Visitor &v);

};

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj);

/// Defines the JSON representation to be used for vector<unique_ptr<AbstractStatement>> objects.
void to_json(json &j, const std::unique_ptr<AbstractStatement> &param);


#endif //MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
