#ifndef AST_OPTIMIZER_INCLUDE_PRINTVISITOR_H
#define AST_OPTIMIZER_INCLUDE_PRINTVISITOR_H

#include "Visitor.h"
#include <list>
#include <sstream>
#include <string>

class PrintVisitor : public Visitor {
protected:
    int level;
    Scope *lastPrintedScope;
    std::stringstream ss;
    bool printScreen;

public:
    PrintVisitor();

    explicit PrintVisitor(bool printScreen);

    virtual ~PrintVisitor();

    void visit(Ast &elem) override;

    void visit(BinaryExpr &elem) override;

    void visit(Block &elem) override;

    void visit(Call &elem) override;

    void visit(CallExternal &elem) override;

    void visit(Function &elem) override;

    void visit(FunctionParameter &elem) override;

    void visit(If &elem) override;

    void visit(LiteralBool &elem) override;

    void visit(LiteralInt &elem) override;

    void visit(LiteralString &elem) override;

    void visit(LiteralFloat &elem) override;

    void visit(LogicalExpr &elem) override;

    void visit(Operator &elem) override;

    void visit(Return &elem) override;

    void visit(UnaryExpr &elem) override;

    void visit(VarAssignm &elem) override;

    void visit(VarDecl &elem) override;

    void visit(Variable &elem) override;

    void visit(While &elem) override;

    void incrementLevel();

    void decrementLevel();

    void resetLevel();

    std::string getIndentation();

    void addOutputStr(AbstractNode &node, const std::list<std::string> &args);

    [[nodiscard]] Scope *getLastPrintedScope() const;

    void setLastPrintedScope(Scope *scope);

    template<typename T>
    void printChildNodesIndented(T &elem);

    std::string getOutput() const;

    void resetVisitor();

    void printScope();

    void addOutputStr(AbstractNode &node);

    void printNodeName(AbstractNode &node);
};

#endif //AST_OPTIMIZER_INCLUDE_PRINTVISITOR_H
