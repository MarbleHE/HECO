
#ifndef MASTER_THESIS_CODE_AST_H
#define MASTER_THESIS_CODE_AST_H


#include "AbstractStatement.h"

class Ast {
private:
    AbstractStatement *rootNode;
public:
    Ast(AbstractStatement *rootNode);

    Ast();

    AbstractStatement *setRootNode(AbstractStatement *rootNode);

    AbstractStatement *getRootNode() const;
};


#endif //MASTER_THESIS_CODE_AST_H
