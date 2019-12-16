#ifndef MASTER_THESIS_CODE_BLOCK_H
#define MASTER_THESIS_CODE_BLOCK_H


#include <vector>
#include "AbstractStatement.h"

class Block : public AbstractStatement, public Node {
private:
    std::vector<AbstractStatement *> *statements;
public:

    Block();

    Block(AbstractStatement *stat);

    Block(std::vector<AbstractStatement *> *statements);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    std::string getNodeName() const override;

    std::vector<AbstractStatement *> *getStatements() const;
};


#endif //MASTER_THESIS_CODE_BLOCK_H
