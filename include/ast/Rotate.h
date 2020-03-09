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
  Rotate(AbstractExpr *vector, int rotationFactor) : rotationFactor(rotationFactor) {
    setAttributes(vector);
  }

  [[nodiscard]] int getRotationFactor() const {
    return rotationFactor;
  }

  int getMaxNumberChildren() override {
    return 1;
  }

  [[nodiscard]] json toJson() const override {
    json j;
    j["type"] = getNodeType();
    j["operand"] = getOperand()->toJson();
    j["rotationFactor"] = getRotationFactor();
    return j;
  }

  [[nodiscard]] std::string toString(bool printChildren) const override {
    return AbstractNode::generateOutputString(printChildren, {std::to_string(getRotationFactor())});
  }

  AbstractNode *cloneFlat() override {
    return new Rotate(nullptr, this->getRotationFactor());
  }

  bool supportsCircuitMode() override {
    return true;
  }

  [[nodiscard]] AbstractExpr *getOperand() const {
    return reinterpret_cast<AbstractExpr *>(getChildAtIndex(0));
  }

  [[nodiscard]] std::string getNodeType() const override {
    return std::string("Rotate");
  }

  void accept(Visitor &v) override {
    v.visit(*this);
  }

  AbstractNode *clone(bool keepOriginalUniqueNodeId) override {
    return new Rotate(this->getChildAtIndex(0)->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                      getRotationFactor());
  }

  void setAttributes(AbstractExpr *pExpr) {
    // Rotation requires either an AbstractLiteral that is a 1-dimensional row or column vector, or a Variable in which
    // case it is not possible at compile-time to determine whether the variable satisfies the former requirement. Must
    // be checked while evaluating the AST.
    if (dynamic_cast<AbstractLiteral *>(pExpr)!=nullptr && !isOneDimensionalVector()) {
      throw std::logic_error("Rotate requires a 1-dimensional row or column vector.");
    } else if (dynamic_cast<Variable *>(pExpr)==nullptr) {
      throw std::logic_error("Rotate is supported for AbstractLiterals and Variables only.");
    }
    removeChildren();
    addChildren({pExpr}, true);
  }

  bool isOneDimensionalVector() {
    Dimension *dim;
    auto expressionToRotate = getOperand();
    if (auto literal = dynamic_cast<AbstractLiteral *>(expressionToRotate)) {
      dim = &(literal->getMatrix()->getDimensions());
      return dim->hasDimension(1, -1) || dim->hasDimension(-1, 1);
    }
    return false;
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_ROTATE_H_
