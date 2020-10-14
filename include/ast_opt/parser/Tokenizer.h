#ifndef tokenizer_hpp
#define tokenizer_hpp

#include <functional>
#include <string_view>
#include <iostream>
#include <deque>

#include "Tokens.h"

namespace stork {
class PushBackStream;

class tokens_iterator {
 private:
  std::function<token()> _get_next_token;
  token _current;

 public:
  explicit tokens_iterator(PushBackStream &stream);

  explicit tokens_iterator(std::deque<token> &tokens);

  tokens_iterator(const tokens_iterator &) = delete;

  void operator=(const tokens_iterator &) = delete;

  const token &operator*() const;

  const token *operator->() const;

  tokens_iterator &operator++();

  explicit operator bool() const;
};
}

#endif /* tokenizer_hpp */
