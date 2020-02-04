#ifndef MASTER_THESIS_CODE_LOGICALEXPR_H
#define MASTER_THESIS_CODE_LOGICALEXPR_H

#include "Operator.h"
#include "AbstractExpr.h"
#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"
#include <string>
#include <vector>

class LogicalExpr : public AbstractExpr {
private:
    AbstractExpr *left;
    Operator *op;
    AbstractExpr *right;

public:
    LogicalExpr();

    explicit LogicalExpr(OpSymb::LogCompOp op);

    template<typename T1, typename T2>
    LogicalExpr(T1 left, OpSymb::LogCompOp op, T2 right) {
        this->left = createParam(left);
        this->op = new Operator(op);
        this->right = createParam(right);
        std::vector<Node *> nodes{this->left, this->op, this->right};
        this->addChildren(nodes);
        Node::addParent(this, nodes);
    }

    ~LogicalExpr() override;

    [[nodiscard]] AbstractExpr *getLeft() const;

    [[nodiscard]] Operator &getOp() const;

    [[nodiscard]] AbstractExpr *getRight() const;

    [[nodiscard]] json toJson() const
    override;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    Literal *evaluate(Ast &ast) override;

    std::vector<std::string> getVariableIdentifiers() override;

    int countByTemplate(AbstractExpr *abstractExpr) override;

    LogicalExpr *contains(LogicalExpr *lexpTemplate, AbstractExpr *excludedSubtree);

    Node *clone() override;

    void setOperands(AbstractExpr *lexp, AbstractExpr *rexp);
};

#endif //MASTER_THESIS_CODE_LOGICALEXPR_H
