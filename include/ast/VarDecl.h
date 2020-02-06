#ifndef MASTER_THESIS_CODE_VARDECL_H
#define MASTER_THESIS_CODE_VARDECL_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"
#include "../utilities/Datatypes.h"

class VarDecl : public AbstractStatement {
 private:
  std::string identifier;

 public:
  VarDecl(std::string name, TYPES datatype, AbstractExpr* initializer);

  VarDecl(std::string name, int value);

  VarDecl(std::string name, bool value);

  /// This is just a helper constructor that allows to call VarDecl("randomString", "aiermkr");
  /// without this constructor the call will be wrongly forwarded to the VarDecl(std::string, bool) constructor.
  /// See https://stackoverflow.com/q/14770252/3017719.
  /// \param name The variable's identifier.
  /// \param valueAssignedTo The value assigned to the variable.
  VarDecl(std::string name, const char* valueAssignedTo);

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
  void setAttributes(std::string varIdentifier, Datatype* datatype, AbstractExpr* value);
  bool supportsCircuitMode() override;
  int getMaxNumberChildren() override;
};

#endif //MASTER_THESIS_CODE_VARDECL_H
