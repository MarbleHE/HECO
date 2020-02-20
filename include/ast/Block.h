#ifndef AST_OPTIMIZER_INCLUDE_BLOCK_H
#define AST_OPTIMIZER_INCLUDE_BLOCK_H

#include <vector>
#include "AbstractStatement.h"
#include <string>

class Block : public AbstractStatement {
 public:
  Block() = default;

  ~Block() override = default;

  Block *clone(bool keepOriginalUniqueNodeId) override;

  AbstractNode *cloneFlat() override;

  explicit Block(AbstractStatement *stat);

  explicit Block(std::vector<AbstractStatement *> *statements);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] std::vector<AbstractStatement *> *getStatements() const;

  int getMaxNumberChildren() override;

  bool supportsCircuitMode() override;

  std::string toString() const override;
};

#endif //AST_OPTIMIZER_INCLUDE_BLOCK_H
