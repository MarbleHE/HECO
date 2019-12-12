
#ifndef MASTER_THESIS_CODE_STATISTICSVISITOR_H
#define MASTER_THESIS_CODE_STATISTICSVISITOR_H

#include "Visitor.h"

class StatisticsVisitor : public Visitor {
public:
    virtual void visit(Ast &elem) override;

    virtual void visit(BinaryExpr &elem) override;

    virtual void visit(Block &elem) override;

    virtual void visit(Call &elem) override;

    virtual void visit(CallExternal &elem) override;

    virtual void visit(Class &elem) override;

    virtual void visit(Function &elem) override;

    virtual void visit(FunctionParameter &elem) override;

    virtual void visit(Group &elem) override;

    virtual void visit(If &elem) override;

    virtual void visit(Literal &elem) override;

    virtual void visit(LiteralBool &elem) override;

    virtual void visit(LiteralInt &elem) override;

    virtual void visit(LiteralString &elem) override;

    virtual void visit(LogicalExpr &elem) override;

    virtual void visit(Operator &elem) override;

    virtual void visit(Return &elem) override;

    virtual void visit(UnaryExpr &elem) override;

    virtual void visit(VarAssignm &elem) override;

    virtual void visit(VarDecl &elem) override;

    virtual void visit(Variable &elem) override;

    virtual void visit(While &elem) override;

};


#endif //MASTER_THESIS_CODE_STATISTICSVISITOR_H
