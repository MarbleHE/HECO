#ifndef MASTER_THESIS_CODE_PRINTVISITOR_H
#define MASTER_THESIS_CODE_PRINTVISITOR_H

#include "Visitor.h"
#include <list>


class PrintVisitor : public Visitor {
protected:
    int level;
    Scope *lastPrintedScope;

public:
    PrintVisitor(const int level);

    virtual ~PrintVisitor();

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

    void incrementLevel();

    void decrementLevel();

    void resetLevel();

    std::string getIndentation();

    std::string formatOutputStr(const std::list<std::string> &args);

    Scope *getLastPrintedScope() const;

    void setLastPrintedScope(Scope *scope);

};


#endif //MASTER_THESIS_CODE_PRINTVISITOR_H
