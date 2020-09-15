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
}

TEST(TokenizerTest, recognizeInputTest) {
  std::string path = __FILE__;
  path = path.substr(0, path.find_last_of("/\\") + 1) + "test.stk";

  stork::file f(path.c_str());

  stork::get_character get = [&]() {
    return f();
  };
  stork::push_back_stream stream(&get);

  stork::tokens_iterator it(stream);

  int iteration_index = 0;
  while(it) {

    std::cout <<  iteration_index << ": " <<  std::to_string(it->get_value()) << std::endl;
    ++iteration_index;
    ++it;
  }

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
