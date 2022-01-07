#include "abc/ast_parser/PushBackStream.h"

namespace stork {
PushBackStream::PushBackStream(const get_character *input) :
    _input(*input),
    _line_number(0),
    _char_index(0) {
}

char PushBackStream::operator()() {
  char ret = -1;
  if (_stack.empty()) {
    ret = _input();
  } else {
    ret = _stack.top();
    _stack.pop();
  }
  if (ret=='\n') {
    ++_line_number;
  }

  ++_char_index;

  return ret;
}

void PushBackStream::pushBack(char c) {
  _stack.push(c);

  if (c=='\n') {
    --_line_number;
  }

  --_char_index;
}

size_t PushBackStream::getLineNumber() const {
  return _line_number;
}

size_t PushBackStream::getCharIndex() const {
  return _char_index;
}
}

