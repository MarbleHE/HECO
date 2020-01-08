#ifndef MASTER_THESIS_CODE_INCLUDE_UTILITIES_DATATYPES_H_
#define MASTER_THESIS_CODE_INCLUDE_UTILITIES_DATATYPES_H_

#include <string>
#include <map>

enum class TYPES {
  INT, FLOAT, STRING, BOOL
};

class Datatype {
 public:
  const TYPES val;
  bool isEncrypted = false;

  explicit Datatype(TYPES di) : val(di) {};
  explicit Datatype(TYPES di, bool isEncrypted) : val(di), isEncrypted(isEncrypted) {};

  std::string enum_to_string(const TYPES identifiers) const {
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
};

#endif //MASTER_THESIS_CODE_INCLUDE_UTILITIES_DATATYPES_H_
