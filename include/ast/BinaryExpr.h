#ifndef MASTER_THESIS_CODE_BINARYEXPR_H
#define MASTER_THESIS_CODE_BINARYEXPR_H

#include "Operator.h"
#include "AbstractExpr.h"
#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"
#include <string>

class BinaryExpr : public AbstractExpr {
 protected:
  AbstractExpr* left;
  Operator* op;
  AbstractExpr* right;

 public:
  /// Represents an expression of the form "left op right", e.g., "2 + a" or "53 * 3".
  /// \param left is the left operand of the expression.
  /// \param op is the operator of the expression.
  /// \param right is the right operand of the expression.
  BinaryExpr(AbstractExpr* left, OpSymb::BinaryOp op, AbstractExpr* right);

  template<typename T1, typename T2>
  BinaryExpr(T1 left, OpSymb::BinaryOp op, T2 right) {
    this->left = createParam(left);
    this->op = new Operator(op);
    this->right = createParam(right);
  }

  ~BinaryExpr() override;

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] AbstractExpr* getLeft() const;

  [[nodiscard]] Operator &getOp() const;

  [[nodiscard]] AbstractExpr* getRight() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  void setLeft(AbstractExpr* value);

  void setOp(Operator* operatore);

  void setRight(AbstractExpr* rhs);

  static void swapOperandsLeftAWithRightB(BinaryExpr* bexpA, BinaryExpr* bexpB);

  explicit BinaryExpr(OpSymb::BinaryOp op);

  BinaryExpr* contains(BinaryExpr* bexpTemplate, AbstractExpr* excludedSubtree) override;

  bool contains(Variable* var) override;

  bool isEqual(AbstractExpr* other) override;

  Literal* evaluate(Ast &ast) override;
};

#endif //MASTER_THESIS_CODE_BINARYEXPR_H
