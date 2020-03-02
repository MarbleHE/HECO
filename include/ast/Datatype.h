#ifndef AST_OPTIMIZER_INCLUDE_AST_DATATYPE_H_
#define AST_OPTIMIZER_INCLUDE_AST_DATATYPE_H_

#include <string>
#include <map>
#include "AbstractNode.h"

enum class Types {
  INT, FLOAT, STRING, BOOL
};

class Datatype : public AbstractNode {
 private:
  Types val;
  bool encrypted{false};

 public:
  [[nodiscard]] std::string getNodeType() const override;

  AbstractNode *cloneFlat() override;

  AbstractNode *clone(bool keepOriginalUniqueNodeId) override;

  explicit Datatype(Types di) : val(di) {}

  explicit Datatype(Types di, bool isEncrypted) : val(di), encrypted(isEncrypted) {}

  explicit Datatype(const std::string &type);

  static std::string enumToString(Types identifiers);

  explicit operator std::string() const;

  explicit operator Types() const;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool operator==(const Datatype &rhs) const;

  bool operator!=(const Datatype &rhs) const;

  [[nodiscard]] Types getType() const;

  [[nodiscard]] bool isEncrypted() const;

  void setEncrypted(bool isEncrypted);

  void accept(Visitor &v) override;

  bool supportsCircuitMode() override;

  [[nodiscard]] json toJson() const override;

  /// Returns the default variable initialization value based on the given datatype. For example, integers (int) by
  /// default are initialized by 0. This method is needed because our VarDecl object allows to leave out the initializer
  /// in which case the variable's value is undefined. For certain actions, e.g., compile-time expression
  /// simplification we need to know the variable's initial value.
  /// \param datatype The datatype for which the default initialization value should be determined.
  /// \return The default value to which an uninitialized (declared only) variable is initialized to.
  static AbstractLiteral *getDefaultVariableInitializationValue(Types datatype);
};

#endif //AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPE_H_
