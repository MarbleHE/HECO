#ifndef MASTER_THESIS_CODE_BLOCK_H
#define MASTER_THESIS_CODE_BLOCK_H


#include <vector>
#include "AbstractStatement.h"

class Block : public AbstractStatement, public Node {
private:
    std::vector<AbstractStatement *> *statements;

public:
    Block();

    explicit Block(AbstractStatement *stat);

    explicit Block(std::vector<AbstractStatement *> *statements);

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    [[nodiscard]] std::vector<AbstractStatement *> *getStatements() const;

    virtual ~Block();
};


#endif //MASTER_THESIS_CODE_BLOCK_H
