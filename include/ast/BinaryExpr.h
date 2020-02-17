#ifndef AST_OPTIMIZER_INCLUDE_BINARYEXPR_H
#define AST_OPTIMIZER_INCLUDE_BINARYEXPR_H

#include <string>
#include <vector>
#include "Operator.h"
#include "AbstractExpr.h"
#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"

class BinaryExpr : public AbstractExpr {
public:
    /// Represents an expression of the form "left op right", e.g., "2 + a" or "53 * 3".
    /// \param left is the left operand of the expression.
    /// \param op is the operator of the expression.
    /// \param right is the right operand of the expression.
    BinaryExpr(AbstractExpr *left, OpSymb::BinaryOp op, AbstractExpr *right);

    BinaryExpr();

    explicit BinaryExpr(OpSymb::BinaryOp op);

    template<typename T1, typename T2>
    BinaryExpr(T1 left, OpSymb::BinaryOp op, T2 right) {
      setAttributes(AbstractExpr::createParam(left), new Operator(op), AbstractExpr::createParam(right));
    }

    template<typename T1, typename T2>
    BinaryExpr(T1 left, Operator *op, T2 right) {
      setAttributes(AbstractExpr::createParam(left), op, AbstractExpr::createParam(right));
    }

    ~BinaryExpr() override;

    [[nodiscard]] json toJson() const override;

    [[nodiscard]] AbstractExpr *getLeft() const;

    [[nodiscard]] Operator *getOp() const;

    [[nodiscard]] AbstractExpr *getRight() const;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    static void swapOperandsLeftAWithRightB(BinaryExpr *bexpA, BinaryExpr *bexpB);

    BinaryExpr *contains(BinaryExpr *bexpTemplate, AbstractExpr *excludedSubtree) override;

    bool contains(Variable *var) override;

    bool isEqual(AbstractExpr *other) override;

    std::vector<Literal *> evaluate(Ast &ast) override;

    int countByTemplate(AbstractExpr *abstractExpr) override;

    std::vector<std::string> getVariableIdentifiers() override;

    void setAttributes(AbstractExpr *leftOperand, Operator *operatore, AbstractExpr *rightOperand);

    int getMaxNumberChildren() override;

    bool supportsCircuitMode() override;

private:
    Node *createClonedNode(bool keepOriginalUniqueNodeId) override;
};

#endif //AST_OPTIMIZER_INCLUDE_BINARYEXPR_H
