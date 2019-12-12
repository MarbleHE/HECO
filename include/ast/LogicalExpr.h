#ifndef MASTER_THESIS_CODE_LOGICALEXPR_H
#define MASTER_THESIS_CODE_LOGICALEXPR_H


#include "Operator.h"
#include "AbstractExpr.h"


class LogicalExpr : public AbstractExpr {
    AbstractExpr *left;
    LogicalCompOperator op;
    AbstractExpr *right;
public:

    LogicalExpr(AbstractExpr *left, LogicalCompOperator op, AbstractExpr *right);

    // TODO implement constructors or find better way to solve that
    // helper constructors that automatically instantiate the underlying objects for common variations
//    LogicalExpr(int literalIntLeft, LogicalCompOperator op, std::string variableRight);
//    LogicalExpr(int literalIntLeft, LogicalCompOperator op, bool literalBoolRight);
//    LogicalExpr(std::string variableLeft, LogicalCompOperator op, int literalIntRight);
//    LogicalExpr(std::string variableLeft, LogicalCompOperator op, bool literalBoolRight);
//    LogicalExpr(bool literalBoolLeft, LogicalCompOperator op, std::string variableRight);
//    LogicalExpr(bool literalBoolLeft, LogicalCompOperator op, int literalIntRight);
//    // -- constructors with same type on lhs and rhs
//    template <typename T>
//    LogicalExpr(T variableLeft, LogicalCompOperator op, T variableRight);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

};


#endif //MASTER_THESIS_CODE_LOGICALEXPR_H
