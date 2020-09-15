#ifndef expression_hpp
#define expression_hpp

#include "Variable.h"
#include "Types.h"

#include <string>

namespace stork {
class runtime_context;
class tokens_iterator;
class compiler_context;

template<typename R>
class expression {
  expression(const expression &) = delete;
  void operator=(const expression &) = delete;
 protected:
  expression() = default;
 public:
  using ptr = std::unique_ptr<const expression>;

  virtual R evaluate(runtime_context &context) const = 0;
  virtual ~expression() = default;
};

expression<void>::ptr build_void_expression(compiler_context &context, tokens_iterator &it);
expression<number>::ptr build_number_expression(compiler_context &context, tokens_iterator &it);
expression<integer>::ptr build_integer_expression(compiler_context &context, tokens_iterator &it);
expression<lvalue>::ptr build_initialization_expression(
    compiler_context &context,
    tokens_iterator &it,
    type_handle type_id,
    bool allow_comma
);
expression<lvalue>::ptr build_default_initialization(type_handle type_id);
}

#endif /* expression_hpp */
