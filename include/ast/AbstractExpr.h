#ifndef MASTER_THESIS_CODE_ABSTRACTEXPR_H
#define MASTER_THESIS_CODE_ABSTRACTEXPR_H

#include <string>


class AbstractExpr {
public:
    virtual ~AbstractExpr() = default;

    virtual std::string toString() const;

};

std::ostream &operator<<(std::ostream &outs, const AbstractExpr &obj);


#endif //MASTER_THESIS_CODE_ABSTRACTEXPR_H
