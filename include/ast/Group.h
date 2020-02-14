#ifndef AST_OPTIMIZER_INCLUDE_GROUP_H
#define AST_OPTIMIZER_INCLUDE_GROUP_H

#include "AbstractExpr.h"
#include <string>

class Group : public AbstractExpr {
 private:
  Node* createClonedNode(bool keepOriginalUniqueNodeId) override;

 public:
  explicit Group(AbstractExpr* expr);

  ~Group() override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] AbstractExpr* getExpr() const;

  BinaryExpr* contains(BinaryExpr* bexpTemplate, AbstractExpr* excludedSubtree) override;

  Literal* evaluate(Ast &ast) override;

  bool supportsCircuitMode() override;

  int getMaxNumberChildren() override;

  void setAttributes(AbstractExpr* expression);
};

#endif //AST_OPTIMIZER_INCLUDE_GROUP_H
