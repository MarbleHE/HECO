#ifndef MASTER_THESIS_CODE_BLOCK_H
#define MASTER_THESIS_CODE_BLOCK_H

#include <vector>
#include "AbstractStatement.h"
#include <string>

class Block : public AbstractStatement {
 private:
  std::vector<AbstractStatement*>* statements;

 public:
  Block();

  ~Block();

  explicit Block(AbstractStatement* stat);

  explicit Block(std::vector<AbstractStatement*>* statements);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] std::vector<AbstractStatement*>* getStatements() const;

  Literal* evaluate(Ast &ast) override;

};

#endif //MASTER_THESIS_CODE_BLOCK_H
