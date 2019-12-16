#ifndef MASTER_THESIS_CODE_LITERALSTRING_H
#define MASTER_THESIS_CODE_LITERALSTRING_H


#include <string>
#include "Literal.h"

class LiteralString : public Literal, public Node {
private:
    std::string value;
public:
    LiteralString(const std::string &value);

    json toJson() const override;

    const std::string &getValue() const;

    virtual void accept(Visitor &v) override;

    std::string getNodeName() const override;
};


#endif //MASTER_THESIS_CODE_LITERALSTRING_H
