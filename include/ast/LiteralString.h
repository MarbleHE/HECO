#ifndef MASTER_THESIS_CODE_LITERALSTRING_H
#define MASTER_THESIS_CODE_LITERALSTRING_H


#include <string>
#include "Literal.h"

class LiteralString : public Literal {
    std::string value;
public:
    LiteralString(const std::string &value);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_LITERALSTRING_H
