#include "abc/ast_parser/Parser.h"
#include "ASTComparison.h"
#include "abc/ast/Variable.h"
#include "gtest/gtest.h"


TEST(VariableTest, values_IdentifierGivenInCtorIsRetrievable) {
  // Variables are little more than a wrapper around a identifier,
  // this test ensures that when we create a Variable with an identifier, we can later retrieve it

  Variable foo("foo");

  EXPECT_EQ(foo.getIdentifier(), "foo");
}

TEST(VariableTest, values_CopyCtorCopiesIdentifier) {
  // When copying a Variable, the new object should contain the same identifier

  Variable foo("foo");
  Variable foo_copy(foo);

  EXPECT_EQ(foo_copy.getIdentifier(), "foo");
}

TEST(VariableTest, values_CopyAssignmentCopiesIdentifier) {
  // When copying a Variable, the new object should contain the same identifier

  Variable foo("foo");
  Variable foo_copy = foo;

  EXPECT_EQ(foo_copy.getIdentifier(), "foo");
}

TEST(VariableTest, values_MoveCtorPreservesIdentifier) {
  // When moving a Variable, the new object should contain the same identifier

  Variable foo("foo");
  Variable new_foo(std::move(foo));

  EXPECT_EQ(new_foo.getIdentifier(), "foo");
}

TEST(VariableTest, values_MoveAssignmentPreservesIdentifier) {
  // When moving a Variable, the new object should contain the same identifier

  Variable foo("foo");
  Variable new_foo("new_foo");
  new_foo = std::move(foo);

  EXPECT_EQ(new_foo.getIdentifier(), "foo");
}

TEST(VariableTest, node_countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  Variable foo("foo");
  auto reported_count = foo.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for(auto &c: foo) {
    c = c; // Use c to suppress warning
    ++actual_count;
  }

  EXPECT_EQ(reported_count,actual_count);
}

TEST(VariableTest, node_VariablessAreLeafNodes) {
  // Variables are supposed to be leaf nodes and should not have any children

  Variable foo("foo");

  EXPECT_EQ(foo.countChildren(),0);
}

TEST(VariableTest, JsonOutputTest) { /* NOLINT */
  std::string varIdentifier = "myVar";
  auto var = new Variable(varIdentifier);
  nlohmann::json j = {{"type", "Variable"},
            {"identifier", varIdentifier}};
  EXPECT_EQ(var->toJson(), j);
}

TEST(VariableTest, JsonInputTest) {
  std::string varIdentifier = "myVar";
  auto var_expected = Variable(varIdentifier);

  std::string var_json = R""""({
    "identifier": "myVar",
    "type": "Variable"
  })"""";

  auto var_parsed = Parser::parseJson(var_json);

  ASSERT_TRUE(compareAST(*var_parsed, var_expected));
}
