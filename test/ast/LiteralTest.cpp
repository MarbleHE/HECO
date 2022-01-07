#include <abc/ast_parser/Parser.h>
#include "ASTComparison.h"
#include "abc/ast/Literal.h"
#include "gtest/gtest.h"

TEST(LiteralTest, typedefCheck) {
  // There are several typedefs of the Literal<T> template for specific Ts
  // This is mostly for readability and legacy support of code written pre-templatization

  // Here, we simply confirm that all the required templates can be instantiated
  // If not, this should give a compiler error already
  LiteralBool b(true);
  LiteralChar c(true);
  LiteralInt i(1);
  LiteralFloat f(1.2f);
  LiteralDouble d(4.5);
  LiteralString s("Hello World!");

  // and that they have the expected type.
  // The internal parentheses are necessary since otherwise the GTEST macro breaks
  EXPECT_TRUE((std::is_same<LiteralBool, Literal<bool>>::value));
  EXPECT_TRUE((std::is_same<LiteralChar, Literal<char>>::value));
  EXPECT_TRUE((std::is_same<LiteralInt, Literal<int>>::value));
  EXPECT_TRUE((std::is_same<LiteralFloat, Literal<float>>::value));
  EXPECT_TRUE((std::is_same<LiteralString, Literal<std::string>>::value));
}

TEST(LiteralTest, values_ValueGivenInCtorIsRetrievable) {
  // Literals are little more than a wrapper around a value,
  // this test ensures that when we create a Literal with a value, we can later retrieve it

  // Since this is not specific to T, we simply use one built-in and one class type here
  Literal<int> i(42);
  Literal<std::string> s("Hello World!");

  EXPECT_EQ(i.getValue(), 42);
  EXPECT_EQ(s.getValue(), "Hello World!");
}

TEST(LiteralTest, values_SetAndGet) {
  // This test simply checks that values can be set and get correctly.

  // Since this is not specific to T, we simply use one built-in and one class type here
  Literal<int> i(42);
  Literal<std::string> s("Hello World!");

  // Set new values
  i.setValue(1);
  s.setValue("Goodbye!");

  EXPECT_EQ(i.getValue(), 1);
  EXPECT_EQ(s.getValue(), "Goodbye!");
}

TEST(LiteralTest, values_CopyCtorCopiesValue) {
  // When copying a literal, the new object should contain a copy of the value

  // Since this is not specific to T, we simply use one built-in and one class type here
  Literal<int> i(42);
  Literal<std::string> s("Hello World!");

  // Copy the literals
  Literal<int> i_copy(i);
  Literal<std::string> s_copy(s);

  EXPECT_EQ(i_copy.getValue(), 42);
  EXPECT_EQ(s_copy.getValue(), "Hello World!");
}

TEST(LiteralTest, values_CopyAssignmentCopiesValue) {
  // When copying a literal, the new object should contain a copy of the value

  // Since this is not specific to T, we simply use one built-in and one class type here
  Literal<int> i(42);
  Literal<std::string> s("Hello World!");

  // Copy the literals
  Literal<int> i_copy(1);
  Literal<std::string> s_copy("Goodbye!");
  i_copy = i;
  s_copy = s;

  EXPECT_EQ(i_copy.getValue(), 42);
  EXPECT_EQ(s_copy.getValue(), "Hello World!");
}

TEST(LiteralTest, values_MoveCtorPreservesValue) {
  // When moving a literal, the new object should contain the same value

  // Since this is not specific to T, we simply use one built-in and one class type here
  Literal<int> i(42);
  Literal<std::string> s("Hello World!");

  // Move the literals
  Literal<int> i_new(std::move(i));
  Literal<std::string> s_new(std::move(s));

  EXPECT_EQ(i_new.getValue(), 42);
  EXPECT_EQ(s_new.getValue(), "Hello World!");
}

TEST(LiteralTest, values_MoveAssignmentPreservesValue) {
  // When moving a literal, the new object should contain the same value

  // Since this is not specific to T, we simply use one built-in and one class type here
  Literal<int> i(42);
  Literal<std::string> s("Hello World!");

  // Move the literals
  Literal<int> i_new(1);
  Literal<std::string> s_new("new");
  i_new = std::move(i);
  s_new = std::move(s);

  EXPECT_EQ(i_new.getValue(), 42);
  EXPECT_EQ(s_new.getValue(), "Hello World!");
}

TEST(LiteralTest, node_countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  Literal<int> i(42);
  auto reported_count = i.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for (auto &c: i) {
    c = c; // Use c to suppress warning
    ++actual_count;
  }

  EXPECT_EQ(reported_count, actual_count);
}

TEST(LiteralTest, node_literalsAreLeafNodes) {
  // Literals are supposed to be leaf nodes and should not have any children

  Literal<int> i(42);

  EXPECT_EQ(i.countChildren(), 0);
}

TEST(LiteralTest, JsonOutputTestLiteralInt) { /* NOLINT */
  int val = 2;
  auto *lint = new LiteralInt(val);
  nlohmann::json j = {{"type", "LiteralInt"},
                      {"value", val}};
  EXPECT_EQ(lint->toJson(), j);
}

TEST(LiteralTest, JsonInputTestLiteralInt) {
  auto expected = LiteralInt(2);

  std::string json = R""""({
    "type": "LiteralInt",
    "value": 2
  })"""";

  auto parsed = Parser::parseJson(json);

  ASSERT_TRUE(compareAST(*parsed, expected));
}

TEST(LiteralTest, JsonOutputTestLiteralFloat) { /* NOLINT */
  float val = 33.214f;
  auto *lint = new LiteralFloat(val);
  nlohmann::json j = {{"type", "LiteralFloat"},
                      {"value", val}};
  EXPECT_EQ(lint->toJson(), j);
}

TEST(LiteralTest, JsonInputTestLiteralFloat) {
  auto expected = LiteralFloat(33.214f);

  std::string json = R""""({
    "type": "LiteralFloat",
    "value": 33.214
  })"""";

  auto parsed = Parser::parseJson(json);

  ASSERT_TRUE(compareAST(*parsed, expected));
}

TEST(LiteralTest, JsonOutputTestLiteralBool) { /* NOLINT */
  bool val = true;
  auto *lbool = new LiteralBool(val);
  nlohmann::json j = {{"type", "LiteralBool"},
                      {"value", val}};
  EXPECT_EQ(lbool->toJson(), j);
}

TEST(LiteralTest, JsonInputTestLiteralBool) {
  auto expected = LiteralBool(true);

  std::string json = R""""({
    "type": "LiteralBool",
    "value": true
  })"""";

  auto parsed = Parser::parseJson(json);

  ASSERT_TRUE(compareAST(*parsed, expected));
}

TEST(LiteralTest, JsonOutputTestLiteralString) { /* NOLINT */
  std::string val = "hello world!";
  auto *lString = new LiteralString(val);
  nlohmann::json j = {{"type", "LiteralString"},
                      {"value", val}};
  EXPECT_EQ(lString->toJson(), j);
}

TEST(LiteralTest, JsonInputTestLiteralString) {
  auto expected = LiteralString("hello world!");

  std::string json = R""""({
    "type": "LiteralString",
    "value": "hello world!"
  })"""";

  auto parsed = Parser::parseJson(json);

  ASSERT_TRUE(compareAST(*parsed, expected));
}

// TODO: test LiteralChar, LiteralDouble