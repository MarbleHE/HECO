#ifndef MASTER_THESIS_CODE_VARDECL_H
#define MASTER_THESIS_CODE_VARDECL_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarDecl : public AbstractStatement {
 private:
  std::string identifier;
  std::string datatype;
  AbstractExpr* initializer;

 public:
  VarDecl(std::string name, std::string datatype);

  VarDecl(std::string name, std::string datatype, AbstractExpr* initializer);

  VarDecl(std::string name, const std::string &datatype, int i);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] const std::string &getIdentifier() const;

  [[nodiscard]] const std::string &getDatatype() const;

  [[nodiscard]] AbstractExpr* getInitializer() const;

  BinaryExpr* contains(BinaryExpr* bexpTemplate, BinaryExpr* excludedSubtree) override;

  ~VarDecl() override;

  std::string getVarTargetIdentifier() override;

  bool isEqual(AbstractStatement* as) override;

  Literal* evaluate(Ast &ast) override;
};

#endif //MASTER_THESIS_CODE_VARDECL_H
