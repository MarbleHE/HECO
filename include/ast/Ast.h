#ifndef MASTER_THESIS_CODE_AST_H
#define MASTER_THESIS_CODE_AST_H

#include "AbstractStatement.h"
#include "../visitor/Visitor.h"

class Ast {
 private:
  AbstractStatement *rootNode;
 public:
  explicit Ast(AbstractStatement *rootNode);

  Ast();

  virtual ~Ast();

  AbstractStatement *setRootNode(AbstractStatement *rootNode);

  [[nodiscard]] AbstractStatement *getRootNode() const;

  virtual void accept(Visitor &v);
};

#endif //MASTER_THESIS_CODE_AST_H
