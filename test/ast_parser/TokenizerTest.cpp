#include <include/heco/ast_parser/File.h>
#include <include/heco/ast_parser/Parser.h>
#include "heco/ast/For.h"
#include "heco/ast/Return.h"
#include "heco/ast/VariableDeclaration.h"
#include "heco/ast/Assignment.h"
#include "heco/ast/If.h"
#include "heco/ast/Function.h"
#include "heco/ast/AbstractStatement.h"
#include "heco/ast_parser/PushBackStream.h"
#include "heco/ast_parser/Tokenizer.h"
#include "heco/ast_parser/Errors.h"
#include "gtest/gtest.h"

using std::to_string;

TEST(TokenizerTest, recognizeInputTest)
{ /* NOLINT */
  const char *demoProgram =
      "public secret int main() { "
      "  return; "
      "} ";
  std::string inputCode(demoProgram);

  std::vector<std::string> expected = {"public", "secret", "int", "main", "(", ")", "{", "return", ";", "}"};

  // Setup Tokenizer from String
  auto getFunc = stork::getCharacterFunc(inputCode);
  stork::PushBackStream stream(&getFunc);
  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it)
  {
    actual.push_back(to_string(it->getValue()));
    ++it;
  }

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(TokenizerTest, floatingPointTest)
{ /* NOLINT */
  std::string s = "5.4";
  std::cout << s << std::endl;

  std::vector<std::string> expected = {to_string(5.4)};

  // Setup Tokenizer from String
  auto getFunc = stork::getCharacterFunc(s);
  stork::PushBackStream stream(&getFunc);
  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it)
  {
    actual.push_back(to_string(it->getValue()));
    ++it;
  }

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(TokenizerTest, integerTest)
{ /* NOLINT */
  std::string s = "5";
  std::cout << s << std::endl;

  std::vector<std::string> expected = {to_string(5)};

  // Setup Tokenizer from String
  auto getFunc = stork::getCharacterFunc(s);
  stork::PushBackStream stream(&getFunc);
  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it)
  {
    actual.push_back(to_string(it->getValue()));
    ++it;
  }
}

TEST(TokenizerTest, fromStringTest)
{ /* NOLINT */
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
       "return", "a", ";", "}"};

  std::cout << s << std::endl;

  // Setup Tokenizer from String
  auto getFunc = stork::getCharacterFunc(s);
  stork::PushBackStream stream(&getFunc);
  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it)
  {
    actual.push_back(to_string(it->getValue()));
    ++it;
  }

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(TokenizerTest, ifTest)
{ /* NOLINT */
  const char *programCode = R""""(
    public int main(int a) {
      if (a > 5) {
        return 1;
      }
      return 0;
    }
    )"""";
  std::string inputCode = std::string(programCode);

  std::vector<std::string> expected =
      {"public", "int", "main", "(", "int", "a", ")", "{",
       "if", "(", "a", ">", "5",
       ")", "{", "return", "1", ";", "}",
       "return", "0", ";", "}"};

  // Setup Tokenizer from String
  auto getFunc = stork::getCharacterFunc(inputCode);
  stork::PushBackStream stream(&getFunc);
  stork::tokens_iterator it(stream);

  std::vector<std::string> actual;
  while (it)
  {
    actual.push_back(to_string(it->getValue()));
    ++it;
  }

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_EQ(actual[i], expected[i]);
  }
}
