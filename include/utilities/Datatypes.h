#ifndef MASTER_THESIS_CODE_INCLUDE_UTILITIES_DATATYPES_H_
#define MASTER_THESIS_CODE_INCLUDE_UTILITIES_DATATYPES_H_

#include <string>
#include <map>

enum class TYPES {
  INT, FLOAT, STRING, BOOL
};

class Datatype : public Node {
 public:
  TYPES val;
  bool isEncrypted = false;

  explicit Datatype(TYPES di) : val(di) {}

  explicit Datatype(TYPES di, bool isEncrypted) : val(di), isEncrypted(isEncrypted) {}

  explicit Datatype(std::string type) {
    static const std::map<std::string, TYPES> string_to_types = {
        {"int", TYPES::INT},
        {"float", TYPES::FLOAT},
        {"string", TYPES::STRING},
        {"bool", TYPES::BOOL}};
    auto result = string_to_types.find(type);
    if (result == string_to_types.end()) throw std::invalid_argument("Unsupported datatype given: " + type);
    val = result->second;
  }

  static std::string enum_to_string(const TYPES identifiers) {
    static const std::map<TYPES, std::string> types_to_string = {
        {TYPES::INT, "int"},
        {TYPES::FLOAT, "float"},
        {TYPES::STRING, "string"},
        {TYPES::BOOL, "bool"}};
    return types_to_string.find(identifiers)->second;
  }

  operator std::string() const {
    return enum_to_string(val);
  }

  operator TYPES() const {
    return val;
  }

  std::string toString() const {
    return enum_to_string(val);
  }

  bool operator==(const Datatype &rhs) const {
    return val == rhs.val &&
        isEncrypted == rhs.isEncrypted;
  }

  bool operator!=(const Datatype &rhs) const {
    return !(rhs == *this);
  }
};

#endif //MASTER_THESIS_CODE_INCLUDE_UTILITIES_DATATYPES_H_
