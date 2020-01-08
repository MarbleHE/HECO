#ifndef MASTER_THESIS_CODE_VARDECL_H
#define MASTER_THESIS_CODE_VARDECL_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"
#include "../utilities/Datatypes.h"

class VarDecl : public AbstractStatement {
 private:
  std::string identifier;
  Datatype* datatype;
  AbstractExpr* initializer;

 public:
  VarDecl(std::string name, const std::string &datatype, AbstractExpr* initializer);
  VarDecl(std::string name, int value);
  VarDecl(std::string name, bool value);
  VarDecl(std::string name, float value);
  VarDecl(std::string name, std::string value);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] const std::string &getIdentifier() const;

  [[nodiscard]] Datatype* getDatatype() const;

  [[nodiscard]] AbstractExpr* getInitializer() const;

  BinaryExpr* contains(BinaryExpr* bexpTemplate, BinaryExpr* excludedSubtree) override;

  ~VarDecl() override;

  std::string getVarTargetIdentifier() override;

  bool isEqual(AbstractStatement* as) override;

  Literal* evaluate(Ast &ast) override;
};

#endif //MASTER_THESIS_CODE_VARDECL_H
