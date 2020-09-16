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

using std::to_string;

TEST(TokenizerTest, recognizeInputTest) {
  std::string path = __FILE__;
  path = path.substr(0, path.find_last_of("/\\") + 1) + "test.stk";

  stork::File f(path.c_str());

  std::vector<std::string> expected =      { "public", "secret", "int", "main", "(", ")", "{", "return", ";", "}"};

  stork::get_character get = [&]() {
    return f();
  };
  stork::PushBackStream stream(&get);

  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it) {
    actual.push_back(to_string(it->getValue()));
    ++it;
  }

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(TokenizerTest, floatingPointTest) {
  std::string s = "5.4";
  std::cout << s << std::endl;

  std::vector<std::string> expected =      {to_string(5.4)};

  // Setup Tokenizer from String
  stork::get_character get = [&s]() {
    if (s.empty()) {
      return (char) EOF;
    } else {
      char c = s.at(0);
      s.erase(0, 1);
      return c;
    }
  };
  stork::PushBackStream stream(&get);
  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it) {
    actual.push_back(to_string(it->getValue()));
    ++it;
  }

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(TokenizerTest, integerTest) {
  std::string s = "5";
  std::cout << s << std::endl;

  std::vector<std::string> expected =      {to_string(5)};

  // Setup Tokenizer from String
  stork::get_character get = [&s]() {
    if (s.empty()) {
      return (char) EOF;
    } else {
      char c = s.at(0);
      s.erase(0, 1);
      return c;
    }
  };
  stork::PushBackStream stream(&get);
  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it) {
    actual.push_back(to_string(it->getValue()));
    ++it;
  }

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
}


TEST(TokenizerTest, fromStringTest) {
  std::string s =
      "public int main() {\n"
      "  int a = 0;\n"
      "  a = a + 5;\n"
      "  return a;\n"
      "}";

  std::vector<std::string> expected =
      {"public", "int", "main", "(", ")", "{",
       "int", "a", "=", "0", ";",
       "a", "=", "a", "+", "5", ";",
       "return", "a", ";", "}"
      };

  std::cout << s << std::endl;

  // Setup Tokenizer from String
  stork::get_character get = [&s]() {
    if(s.empty()) {
      return (char) EOF;
    } else {
      char c = s.at(0);
      s.erase(0,1);
      return c;
    }
  };
  stork::PushBackStream stream(&get);
  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it) {
   actual.push_back(to_string(it->getValue()));
   ++it;
  }

  ASSERT_EQ(actual.size(),expected.size());
  for(size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i],expected[i]);
  }
}
