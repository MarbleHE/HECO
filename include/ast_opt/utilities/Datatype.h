#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_DATATYPE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_DATATYPE_H_

#include <string>
#include <map>

/// Simple ENUM to list the supported types
enum class Type {
  BOOL, CHAR, INT, FLOAT, DOUBLE, STRING,
};

/// String representation of enums
std::string enumToString(const Type type);

/// A Datatype consists of a Type and a secret? flag
class Datatype {
 private:
  /// The underlying type (e.g. int, float)
  Type type;

  /// If this is a secret/encrypted version
  bool isSecret;

 public:

  /// Create a Datatype with underlying type and secret-ness
  /// \param type Underlying type
  /// \param isSecret (optional) is this a secret/encrypted type?
  explicit Datatype(Type type, bool isSecret = false);

  /// Two types are equal iff they have the same underlying type and isSecret flag
  /// \param rhs Type to compare against
  /// \return true iff underlying type and isSecret flag agree
  bool operator==(const Datatype &rhs) const;

  /// Two types are unequal unless they have the same underlying type and isSecret flag
  /// \param rhs Type to compare against
  /// \return false iff underlying type and isSecret flag agree
  bool operator!=(const Datatype &rhs) const;

  /// The underlying type
  /// \return Underlying type
  [[nodiscard]] Type getType() const;

  /// The isSecret flag indicating whether or not this is a secret/encrypted Datatype
  /// \return true if Datatype is secret/encrypted
  [[nodiscard]] bool getSecretFlag() const;

  /// String representation
  /// \return String representing the type, e.g. "secret int" or "int"
  std::string toString() const;

};

#endif //AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPE_H_
