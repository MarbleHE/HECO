#include "ast_opt/parser/Tokenizer.h"
#include "gtest/gtest.h"



auto empty_params = std::vector<std::unique_ptr<FunctionParameter>>();


TEST(TokenizerTest, recognizeInputTest) {
  std::string path = __FILE__;
  path = path.substr(0, path.find_last_of("/\\") + 1) + "test.stk";

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
