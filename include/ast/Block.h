
#ifndef MASTER_THESIS_CODE_BLOCK_H
#define MASTER_THESIS_CODE_BLOCK_H


#include <vector>
#include "AbstractStatement.h"

class Block : public AbstractStatement {
public:

    std::vector<std::unique_ptr<AbstractStatement>> blockStatements;

    Block();

    Block(std::vector<std::unique_ptr<AbstractStatement>> stat);

    void addStatement(std::unique_ptr<AbstractStatement> &&statement);

    json toJson() const;

};


#endif //MASTER_THESIS_CODE_BLOCK_H
