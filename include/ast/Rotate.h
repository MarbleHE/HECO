#ifndef AST_OPTIMIZER_INCLUDE_AST_ROTATE_H_
#define AST_OPTIMIZER_INCLUDE_AST_ROTATE_H_

#include <string>
#include "AbstractExpr.h"
#include "AbstractLiteral.h"
#include "LiteralBool.h"
#include "LiteralString.h"
#include "LiteralFloat.h"
#include "LiteralInt.h"
#include "Variable.h"

class Rotate : public AbstractExpr {
 protected:
  int rotationFactor;

 public:
  Rotate(AbstractExpr *vector, int rotationFactor);

  [[nodiscard]] int getRotationFactor() const;

  int getMaxNumberChildren() override;

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  AbstractNode *cloneFlat() override;

  bool supportsCircuitMode() override;

  [[nodiscard]] AbstractExpr *getOperand() const;

  [[nodiscard]] std::string getNodeType() const override;

  void accept(Visitor &v) override;

  Rotate *clone(bool keepOriginalUniqueNodeId) override;

  void setAttributes(AbstractExpr *pExpr);

  bool isOneDimensionalVector();
};

#endif //AST_OPTIMIZER_INCLUDE_AST_ROTATE_H_
