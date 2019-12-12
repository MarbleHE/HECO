#ifndef MASTER_THESIS_CODE_BLOCK_H
#define MASTER_THESIS_CODE_BLOCK_H


#include <vector>
#include "AbstractStatement.h"

class Block : public AbstractStatement {
public:

    std::vector<std::unique_ptr<AbstractStatement>> blockStatements;

    std::vector<AbstractStatement *> *statements;


    Block();

    Block(AbstractStatement *stat);

    Block(std::vector<AbstractStatement *> *statements);

    void addStatement(std::unique_ptr<AbstractStatement> &&statement);

    json toJson() const;

};


#endif //MASTER_THESIS_CODE_BLOCK_H
