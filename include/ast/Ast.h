
#ifndef MASTER_THESIS_CODE_AST_H
#define MASTER_THESIS_CODE_AST_H


#include "AbstractStatement.h"

class Ast {
public:
    AbstractStatement *rootNode;

    void addRootNode(AbstractStatement *root);
};


#endif //MASTER_THESIS_CODE_AST_H
