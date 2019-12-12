#ifndef MASTER_THESIS_CODE_LITERALINT_H
#define MASTER_THESIS_CODE_LITERALINT_H


#include "Literal.h"

class LiteralInt : public Literal {
private:
    int value;
public:
    int getValue() const;

    LiteralInt(int value);

    json toJson() const override;

    void accept(Visitor &v) override;
};


#endif //MASTER_THESIS_CODE_LITERALINT_H
