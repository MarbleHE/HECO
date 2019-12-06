#ifndef MASTER_THESIS_CODE_ABSTRACTEXPR_H
#define MASTER_THESIS_CODE_ABSTRACTEXPR_H


class AbstractExpr {
public:
    virtual ~AbstractExpr() = default;

    virtual void print();
    // TODO implement interface 'Visitor'
};


#endif //MASTER_THESIS_CODE_ABSTRACTEXPR_H
