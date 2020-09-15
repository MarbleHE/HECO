#include <include/ast_opt/parser/File.h>
#include <include/ast_opt/parser/Parser.h>
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


TEST(TokenizerTest, recognizeInputTest) {
  std::string path = __FILE__;
  path = path.substr(0, path.find_last_of("/\\") + 1) + "test.stk";

  stork::File f(path.c_str());

  stork::get_character get = [&]() {
    return f();
  };
  stork::PushBackStream stream(&get);

  stork::tokens_iterator it(stream);

//  int iteration_index = 0;
//  while (it) {
//
//    std::cout << iteration_index << ": " << std::to_string(it->getValue()) << std::endl;
//    ++iteration_index;
//    ++it;
//  }

  // We enforce a lack of global variables and a single statement (e.g. function) for now:
  Parser().parseStatement(it);

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
