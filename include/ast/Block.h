
#ifndef MASTER_THESIS_CODE_BLOCK_H
#define MASTER_THESIS_CODE_BLOCK_H


#include <vector>
#include "AbstractStatement.h"

class Block : public AbstractStatement {
public:
    std::vector<std::unique_ptr<AbstractStatement>> *statements;
    Block(std::vector<std::unique_ptr<AbstractStatement>> *statements);


};


#endif //MASTER_THESIS_CODE_BLOCK_H
