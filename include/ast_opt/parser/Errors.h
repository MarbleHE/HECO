#ifndef errors_hpp
#define errors_hpp

#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <ostream>

namespace stork {
class Error : public std::exception {
 private:
  std::string _message;
  size_t _line_number;
  size_t _char_index;
 public:
  Error(std::string message, size_t line_number, size_t char_index) noexcept;

  [[nodiscard]] const char *what() const noexcept override;
  [[nodiscard]] size_t getLineNumber() const noexcept;
  [[nodiscard]] size_t getCharIndex() const noexcept;
};

Error throwParsingError(std::string_view message, size_t line_number, size_t char_index);
Error throwSyntaxError(std::string_view message, size_t line_number, size_t char_index);
Error throwSemanticError(std::string_view message, size_t line_number, size_t char_index);
Error throwCompilerError(std::string_view message, size_t line_number, size_t char_index);

Error throwUnexpectedError(std::string_view unexpected, size_t line_number, size_t char_index);
Error throwUnexpectedSyntaxError(std::string_view unexpected, size_t line_number, size_t char_index);
Error throwExpectedSyntaxError(std::string_view expected, size_t line_number, size_t char_index);
Error throwUndeclaredError(std::string_view undeclared, size_t line_number, size_t char_index);
Error throwWrongTypeError(std::string_view source, std::string_view destination, bool lvalue,
                          size_t line_number, size_t char_index);
Error throwAlreadyDeclaredError(std::string_view name, size_t line_number, size_t char_index);

using get_character = std::function<char()>;
void formatError(const Error &err, const get_character &source, std::ostream &output);

class runtime_error : public std::exception {
 private:
  std::string _message;
 public:
  explicit runtime_error(std::string message) noexcept;

  [[nodiscard]] const char *what() const noexcept override;
};

void runtime_assertion(bool b, const char *message);

class FileNotFound : public std::exception {
 private:
  std::string _message;
 public:
  explicit FileNotFound(std::string message) noexcept;

  [[nodiscard]] const char *what() const noexcept override;
};
};

#endif /* errors_hpp */
