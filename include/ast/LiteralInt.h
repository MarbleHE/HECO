
#ifndef MASTER_THESIS_CODE_LITERALINT_H
#define MASTER_THESIS_CODE_LITERALINT_H


#include "Literal.h"

class LiteralInt : public Literal {
private:
    int value;

public:
    LiteralInt(int value);

    std::string toString() const;
};


#endif //MASTER_THESIS_CODE_LITERALINT_H
