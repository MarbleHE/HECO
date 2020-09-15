#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/VariableAssignment.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/parser/PushBackStream.h"
#include "ast_opt/parser/Tokenizer.h"
#include "ast_opt/parser/Errors.h"
#include "gtest/gtest.h"

namespace stork {
class file {
  file(const file &) = delete;
  void operator=(const file &) = delete;
 private:
  FILE *_fp;
 public:
  file(const char *path) :
      _fp(fopen(path, "rt")) {
    if (!_fp) {
      throw file_not_found(std::string("'") + path + "' not found");
    }
  }

  ~file() {
    if (_fp) {
      fclose(_fp);
    }
  }

  int operator()() {
    return fgetc(_fp);
  }
};

/// consume token "value" and throw error if something different
void parse_token_value(tokens_iterator &it, const token_value &value) {
  if (it->has_value(value)) {
    ++it;
    return;
  }
  throw expected_syntax_error(std::to_string(value), it->get_line_number(), it->get_char_index());
}

Function* compile_function_statement(tokens_iterator &it) {
  return nullptr;
}
For* compile_for_statement(tokens_iterator &it) {
  return nullptr;
}
If* compile_if_statement(tokens_iterator &it) {
  return nullptr;
}
Return* compile_return_statement(tokens_iterator &it) {
  return nullptr;
}
Block* compile_block_statement(tokens_iterator &it) {
  return nullptr;
}
VariableDeclaration* compile_variable_declaration_statement(tokens_iterator &it) {
  return nullptr;
}
VariableAssignment* compile_variable_assignment_statement(tokens_iterator &it) {
  return nullptr;
}

///
AbstractStatement* compile_statement(tokens_iterator &it) {
  if (it->is_reserved_token()) {
    switch (it->get_reserved_token()) {
      case reserved_token::kw_for:return compile_for_statement(it);
      case reserved_token::kw_if:return compile_if_statement(it);
      case reserved_token::kw_return:return compile_return_statement(it);
      case reserved_token::open_curly:return compile_block_statement(it);
      case reserved_token::kw_public: return compile_function_statement(it);
      default:
        // it starts with a type?
        return compile_variable_declaration_statement(it);
    }
  } else {
    // it better start with an identifier and be an assignment:
    return compile_variable_assignment_statement(it);
  }
}



} //namespace stork

TEST(TokenizerTest, recognizeInputTest) {
  std::string path = __FILE__;
  path = path.substr(0, path.find_last_of("/\\") + 1) + "test.stk";

  stork::file f(path.c_str());

  stork::get_character get = [&]() {
    return f();
  };
  stork::push_back_stream stream(&get);

  stork::tokens_iterator it(stream);

//  int iteration_index = 0;
//  while (it) {
//
//    std::cout << iteration_index << ": " << std::to_string(it->get_value()) << std::endl;
//    ++iteration_index;
//    ++it;
//  }


  // We enforce a lack of global variables and a single statement (e.g. function) for now:
  compile_statement(it);

//  for (const std::pair<std::string, function> &p : external_functions) {
//    stork::get_character get = [i = 0, &p]() mutable {
//      if (i < p.first.size()) {
//        return int(p.first[i++]);
//      } else {
//        return -1;
//      }
//    };

//    stork::push_back_stream stream(&get);

//    stork::tokens_iterator function_it(stream);


//  using namespace stork;
//
//  module m;
//
//  add_standard_functions(m);
//
//  auto s_main = m.create_public_function_caller<void>("main");
//
//  if (m.try_load(path.c_str(), &std::cerr)) {
//    s_main();
//  }

}
