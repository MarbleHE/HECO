#ifndef AST_OPTIMIZER_INCLUDE_LOGICALEXPR_H
#define AST_OPTIMIZER_INCLUDE_LOGICALEXPR_H

#include "Operator.h"
#include "AbstractExpr.h"
#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"
#include <string>
#include <vector>

class LogicalExpr : public AbstractExpr {
public:
    LogicalExpr();

    explicit LogicalExpr(OpSymb::LogCompOp op);

    template<typename T1, typename T2>
    LogicalExpr(T1 left, OpSymb::LogCompOp op, T2 right) {
      setAttributes(AbstractExpr::createParam(left), new Operator(op), AbstractExpr::createParam(right));
    }

    template<typename T1, typename T2>
    LogicalExpr(T1 left, Operator *op, T2 right) {
      setAttributes(AbstractExpr::createParam(left), op, AbstractExpr::createParam(right));
    }

    [[nodiscard]] AbstractExpr *getLeft() const;

    [[nodiscard]] Operator *getOp() const;

    [[nodiscard]] AbstractExpr *getRight() const;

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    std::vector<std::string> getVariableIdentifiers() override;

    int countByTemplate(AbstractExpr *abstractExpr) override;

    LogicalExpr *contains(LogicalExpr *lexpTemplate, AbstractExpr *excludedSubtree);

    AbstractNode *cloneFlat() override;

    void setAttributes(AbstractExpr *leftOperand, Operator *operatore, AbstractExpr *rightOperand);

protected:
    int getMaxNumberChildren() override;

    bool supportsCircuitMode() override;

private:
    AbstractNode *createClonedNode(bool keepOriginalUniqueNodeId) override;
};

#endif //AST_OPTIMIZER_INCLUDE_LOGICALEXPR_H
