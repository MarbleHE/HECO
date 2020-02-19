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
  [[nodiscard]] std::string getNodeName() const override;

  AbstractNode *cloneFlat() override;

  AbstractNode *clone(bool keepOriginalUniqueNodeId) override;

  explicit Datatype(Types di) : val(di) {}

  explicit Datatype(Types di, bool isEncrypted) : val(di), encrypted(isEncrypted) {}

  explicit Datatype(const std::string &type);

  static std::string enumToString(Types identifiers);

  explicit operator std::string() const;

  explicit operator Types() const;

  [[nodiscard]] std::string toString() const override;

  bool operator==(const Datatype &rhs) const;

  bool operator!=(const Datatype &rhs) const;

  [[nodiscard]] Types getType() const;

  [[nodiscard]] bool isEncrypted() const;

  void setEncrypted(bool encrypted);

  void accept(Visitor &v) override;
};

#endif //AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPE_H_
