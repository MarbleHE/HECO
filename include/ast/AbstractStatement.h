#ifndef MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
#define MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H


#include <string>

class AbstractStatement {
public:
    // TODO implement interface 'Visitor'
    virtual ~AbstractStatement() = default;

    virtual void print();
};


#endif //MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
