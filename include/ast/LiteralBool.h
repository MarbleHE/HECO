
#ifndef MASTER_THESIS_CODE_LITERALBOOL_H
#define MASTER_THESIS_CODE_LITERALBOOL_H


#include "Literal.h"

class LiteralBool : public Literal {
    bool value;
public:
    explicit LiteralBool(bool value);
};


#endif //MASTER_THESIS_CODE_LITERALBOOL_H
