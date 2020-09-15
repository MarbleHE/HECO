#include "ast_opt/parser/Errors.h"
#include <sstream>

namespace stork {
Error::Error(std::string message, size_t line_number, size_t char_index) noexcept:
    _message(std::move(message)),
    _line_number(line_number),
    _char_index(char_index) {
}

const char *Error::what() const noexcept {
  return _message.c_str();
}

size_t Error::getLineNumber() const noexcept {
  return _line_number;
}

size_t Error::getCharIndex() const noexcept {
  return _char_index;
}

Error throwParsingError(std::string_view message, size_t line_number, size_t char_index) {
  std::string error_message("Parsing error: ");
  error_message += message;
  return Error(std::move(error_message), line_number, char_index);
}

Error throwSyntaxError(std::string_view message, size_t line_number, size_t char_index) {
  std::string error_message("Syntax error: ");
  error_message += message;
  return Error(std::move(error_message), line_number, char_index);
}

Error throwSemanticError(std::string_view message, size_t line_number, size_t char_index) {
  std::string error_message("Semantic error: ");
  error_message += message;
  return Error(std::move(error_message), line_number, char_index);
}

Error throwCompilerError(std::string_view message, size_t line_number, size_t char_index) {
  std::string error_message("Compiler error: ");
  error_message += message;
  return Error(std::move(error_message), line_number, char_index);
}

Error throwUnexpectedError(std::string_view unexpected, size_t line_number, size_t char_index) {
  std::string message("Unexpected '");
  message += unexpected;
  message += "'";
  return throwParsingError(message, line_number, char_index);
}

Error throwUnexpectedSyntaxError(std::string_view unexpected, size_t line_number, size_t char_index) {
  std::string message("Unexpected '");
  message += unexpected;
  message += "'";
  return throwSyntaxError(message, line_number, char_index);
}

Error throwExpectedSyntaxError(std::string_view expected, size_t line_number, size_t char_index) {
  std::string message("Expected '");
  message += expected;
  message += "'";
  return throwSyntaxError(message, line_number, char_index);
}

Error throwUndeclaredError(std::string_view undeclared, size_t line_number, size_t char_index) {
  std::string message("Undeclared identifier '");
  message += undeclared;
  message += "'";
  return throwSemanticError(message, line_number, char_index);
}

Error throwWrongTypeError(std::string_view source, std::string_view destination,
                          bool lvalue, size_t line_number,
                          size_t char_index) {
  std::string message;
  if (lvalue) {
    message += "'";
    message += source;
    message += "' is not a lvalue";
  } else {
    message += "Cannot convert '";
    message += source;
    message += "' to '";
    message += destination;
    message += "'";
  }
  return throwSemanticError(message, line_number, char_index);
}

Error throwAlreadyDeclaredError(std::string_view name, size_t line_number, size_t char_index) {
  std::string message = "'";
  message += name;
  message += "' is already declared";
  return throwSemanticError(message, line_number, char_index);
}

void formatError(const Error &err, const get_character &source, std::ostream &output) {
  output << "(" << (err.getLineNumber() + 1) << ") " << err.what() << std::endl;

  size_t char_index = 0;

  for (size_t line_number = 0; line_number < err.getLineNumber(); ++char_index) {
    int c = source();
    if (c < 0) {
      return;
    } else if (c=='\n') {
      ++line_number;
    }
  }

  size_t index_in_line = err.getCharIndex() - char_index;

  std::string line;
  for (size_t idx = 0;; ++idx) {
    int c = source();
    if (c < 0 || c=='\n' || c=='\r') {
      break;
    }
    line += char(c=='\t' ? ' ' : c);
  }

  output << line << std::endl;

  for (size_t idx = 0; idx < index_in_line; ++idx) {
    output << " ";
  }

  output << "^" << std::endl;
}

runtime_error::runtime_error(std::string message) noexcept:
    _message(std::move(message)) {
}

const char *runtime_error::what() const noexcept {
  return _message.c_str();
}

void runtime_assertion(bool b, const char *message) {
  if (!b) {
    throw runtime_error(message);
  }
}

FileNotFound::FileNotFound(std::string message) noexcept:
    _message(std::move(message)) {
}

const char *FileNotFound::what() const noexcept {
  return _message.c_str();
}
}
