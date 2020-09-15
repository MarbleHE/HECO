#include "ast_opt/parser/Tokenizer.h"
#include <map>
#include <string>
#include <cctype>
#include <stack>
#include <cstdlib>
#include "ast_opt/parser/PushBackStream.h"
#include "ast_opt/parser/Errors.h"

namespace stork {
namespace {
enum struct character_type {
  eof,
  space,
  alphanum,
  punct,
};

character_type get_character_type(int c) {
  if (c < 0) {
    return character_type::eof;
  }
  if (std::isspace(c)) {
    return character_type::space;
  }
  if (std::isalpha(c) || std::isdigit(c) || c=='_') {
    return character_type::alphanum;
  }
  return character_type::punct;
}

token fetch_word(PushBackStream &stream) {
  size_t line_number = stream.getLineNumber();
  size_t char_index = stream.getCharIndex();

  std::string word;

  char c = stream();

  bool is_number = isdigit(c);

  do {
    word.push_back(char(c));
    c = stream();

    if (c=='.' && word.back()=='.') {
      stream.pushBack(word.back());
      word.pop_back();
      break;
    }
  } while (get_character_type(c)==character_type::alphanum || (is_number && c=='.'));

  stream.pushBack(c);

  if (std::optional<reservedTokens> t = getKeyword(word)) {
    return token(*t, line_number, char_index);
  } else {
    if (std::isdigit(word.front())) {
      char *endptr;
      // TODO: consider different data types, e.g., float, double, int
      double num = strtol(word.c_str(), &endptr, 0);
      if (*endptr!=0) {
        num = strtod(word.c_str(), &endptr);
        if (*endptr!=0) {
          size_t remaining = word.size() - (endptr - word.c_str());
          throw unexpectedError(
              std::string(1, char(*endptr)),
              stream.getLineNumber(),
              stream.getCharIndex() - remaining
          );
        }
      }
      return token(num, line_number, char_index);
    } else {
      return token(identifier{std::move(word)}, line_number, char_index);
    }
  }
}

token fetch_operator(PushBackStream &stream) {
  size_t line_number = stream.getLineNumber();
  size_t char_index = stream.getCharIndex();

  if (std::optional<reservedTokens> t = getOperator(stream)) {
    return token(*t, line_number, char_index);
  } else {
    std::string unexpected;
    size_t err_line_number = stream.getLineNumber();
    size_t err_char_index = stream.getCharIndex();
    for (int c = stream(); get_character_type(c)==character_type::punct; c = stream()) {
      unexpected.push_back(char(c));
    }
    throw unexpectedError(unexpected, err_line_number, err_char_index);
  }
}

token fetch_string(PushBackStream &stream) {
  size_t line_number = stream.getLineNumber();
  size_t char_index = stream.getCharIndex();

  std::string str;

  bool escaped = false;
  char c = stream();
  for (; get_character_type(c)!=character_type::eof; c = stream()) {
    if (c=='\\') {
      escaped = true;
    } else {
      if (escaped) {
        switch (c) {
          case 't': str.push_back('\t');
            break;
          case 'n': str.push_back('\n');
            break;
          case 'r': str.push_back('\r');
            break;
          case '0': str.push_back('\0');
            break;
          default: str.push_back(c);
            break;
        }
        escaped = false;
      } else {
        switch (c) {
          case '\t':
          case '\n':
          case '\r': stream.pushBack(c);
            throw parsingError("Expected closing '\"'", stream.getLineNumber(), stream.getCharIndex());
          case '"': return token(std::move(str), line_number, char_index);
          default: str.push_back(c);
        }
      }
    }
  }
  stream.pushBack(c);
  throw parsingError("Expected closing '\"'", stream.getLineNumber(), stream.getCharIndex());
}

void skip_line_comment(PushBackStream &stream) {
  char c;
  do {
    c = stream();
  } while (c!='\n' && get_character_type(c)!=character_type::eof);

  if (c!='\n') {
    stream.pushBack(c);
  }
}

void skip_block_comment(PushBackStream &stream) {
  bool closing = false;
  char c;
  do {
    c = stream();
    if (closing && c=='/') {
      return;
    }
    closing = (c=='*');
  } while (get_character_type(c)!=character_type::eof);

  stream.pushBack(c);
  throw parsingError("Expected closing '*/'", stream.getLineNumber(), stream.getCharIndex());
}

token tokenize(PushBackStream &stream) {
  while (true) {
    size_t line_number = stream.getLineNumber();
    size_t char_index = stream.getCharIndex();
    char c = stream();
    switch (get_character_type(c)) {
      case character_type::eof: return {eof(), line_number, char_index};
      case character_type::space: continue;
      case character_type::alphanum: stream.pushBack(c);
        return fetch_word(stream);
      case character_type::punct:
        switch (c) {
          case '"': return fetch_string(stream);
          case '/': {
            char c1 = stream();
            switch (c1) {
              case '/': skip_line_comment(stream);
                continue;
              case '*': skip_block_comment(stream);
                continue;
              default: stream.pushBack(c1);
            }
          }
          default: stream.pushBack(c);
            return fetch_operator(stream);
        }
        break;
    }
  }
}
}

tokens_iterator::tokens_iterator(PushBackStream &stream) :
    _current(eof(), 0, 0),
    _get_next_token([&stream]() {
      return tokenize(stream);
    }) {
  ++(*this);
}

tokens_iterator::tokens_iterator(std::deque<token> &tokens) :
    _current(eof(), 0, 0),
    _get_next_token([&tokens]() {
      if (tokens.empty()) {
        return token(eof(), 0, 0);
      } else {
        token ret = std::move(tokens.front());
        tokens.pop_front();
        return ret;
      }
    }) {
  ++(*this);
}

tokens_iterator &tokens_iterator::operator++() {
  _current = _get_next_token();
  return *this;
}

const token &tokens_iterator::operator*() const {
  return _current;
}

const token *tokens_iterator::operator->() const {
  return &_current;
}

tokens_iterator::operator bool() const {
  return !_current.isEof();
}
}

