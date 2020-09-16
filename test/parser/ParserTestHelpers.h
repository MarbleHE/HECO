#ifndef AST_OPTIMIZER_TEST_PARSER_PARSERTESTHELPERS_CPP_
#define AST_OPTIMIZER_TEST_PARSER_PARSERTESTHELPERS_CPP_

#include <functional>
#include <string>

namespace stork {

using get_character = std::function<char()>;

inline stork::get_character getCharacterFunc(std::string &inputString) {
  return [&inputString]() {
    if (inputString.empty()) {
      return (char) EOF;
    } else {
      char c = inputString.at(0);
      inputString.erase(0, 1);
      return c;
    }
  };
}
}

#endif //AST_OPTIMIZER_TEST_PARSER_PARSERTESTHELPERS_CPP_
