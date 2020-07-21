#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "gtest/gtest.h"

TEST(VariableDeclarationTest, values_ValuesGivenInCtorAreRetrievable) {
  // VariableDeclarations are created with a target Variable and value AbstractExpression and type
  // This test simply confirms that they are retrievable later

  VariableDeclaration declaration(Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  ASSERT_TRUE(declaration.hasTarget());
  EXPECT_EQ(declaration.getTarget().getIdentifier(), "foo");
  EXPECT_EQ(declaration.getDatatype(), Datatype(Type::BOOL));
  ASSERT_TRUE(declaration.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(declaration.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableDeclarationTest, target_and_value_SetAndGet) {
  // This test simply checks that target and value can be set and get correctly.

  VariableDeclaration declaration(Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Set new target and value
  declaration.setTarget(std::make_unique<Variable>("boo"));
  declaration.setValue(std::make_unique<LiteralBool>(false));

  ASSERT_TRUE(declaration.hasTarget());
  EXPECT_EQ(declaration.getTarget().getIdentifier(), "boo");
  EXPECT_EQ(declaration.getDatatype(), Datatype(Type::BOOL));
  ASSERT_TRUE(declaration.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(declaration.getValue()).getValue(), false);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableDeclarationTest, values_CopyCtorCopiesValue) {
  // When copying a VariableDeclaration, the new object should contain a (deep) copy of the target and value

  VariableDeclaration declaration(Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Copy the declaration
  VariableDeclaration declaration_copy(declaration);

  ASSERT_TRUE(declaration_copy.hasTarget());
  EXPECT_EQ(declaration_copy.getTarget().getIdentifier(), "foo");
  EXPECT_EQ(declaration.getDatatype(), Datatype(Type::BOOL));
  ASSERT_TRUE(declaration_copy.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(declaration_copy.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableDeclarationTest, values_CopyAssignmentCopiesValue) {
  // When copying a VariableDeclaration, the new object should contain a copy of the value

  VariableDeclaration declaration(Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Copy the declaration
  VariableDeclaration declaration_copy(Datatype(Type::INT), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(false));
  declaration_copy = declaration;

  ASSERT_TRUE(declaration_copy.hasTarget());
  EXPECT_EQ(declaration_copy.getTarget().getIdentifier(), "foo");
  EXPECT_EQ(declaration_copy.getDatatype(), Datatype(Type::BOOL));
  ASSERT_TRUE(declaration_copy.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(declaration_copy.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableDeclarationTest, values_MoveCtorPreservesValue) {
  // When moving a VariableDeclaration, the new object should contain the same target and value

  VariableDeclaration declaration(Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Move the declaration
  VariableDeclaration new_declaration(std::move(declaration));

  ASSERT_TRUE(new_declaration.hasTarget());
  EXPECT_EQ(new_declaration.getTarget().getIdentifier(), "foo");
  EXPECT_EQ(new_declaration.getDatatype(), Datatype(Type::BOOL));
  ASSERT_TRUE(new_declaration.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(new_declaration.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableDeclarationTest, values_MoveAssignmentPreservesValue) {
  // When moving a VariableDeclaration, the new object should contain the same target and value

  VariableDeclaration declaration(Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Move the declaration
  VariableDeclaration new_declaration(Datatype(Type::INT), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(false));
  new_declaration = std::move(declaration);

  ASSERT_TRUE(new_declaration.hasTarget());
  EXPECT_EQ(new_declaration.getTarget().getIdentifier(), "foo");
  EXPECT_EQ(new_declaration.getDatatype(), Datatype(Type::BOOL));
  ASSERT_TRUE(new_declaration.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(new_declaration.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableDeclarationTest, node_countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  VariableDeclaration declaration(Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));
  auto reported_count = declaration.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for (auto &c: declaration) {
    c = c; // Use c to suppress warning
    ++actual_count;
  }

  EXPECT_EQ(reported_count, actual_count);
}

TEST(VariableDeclarationTest, node_iterate_children) {
  // This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

  VariableDeclaration declaration(Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));
  AbstractNode::iterator it = declaration.begin();
  AbstractNode &variable = *it;
  ++it;
  AbstractNode &value = *it;
  ++it;

  // Iterator should now be at end. Dynamic casts will fail if wrong type.
  EXPECT_EQ(it, declaration.end());
  EXPECT_EQ(dynamic_cast<Variable &>(variable).getIdentifier(), "foo");
  EXPECT_EQ(dynamic_cast<LiteralBool &>(value).getValue(), true);
}

TEST(VariableDeclarationTest, JsonOutputTest) { /* NOLINT */
  auto identifier = "foo";
  auto datatype = Datatype(Type::INT);
  int initializer = 3;
  auto *var = new VariableDeclaration(datatype,
                                      std::make_unique<Variable>(identifier),
                                      std::make_unique<LiteralInt>(initializer));
  nlohmann::json j ={
      {"type", "VariableDeclaration"},
      {"datatype", datatype.toString()},
      {"target", {
        {"type", "Variable"},
        {"identifier", identifier}}},
      {"value", {
              {"type", "LiteralInt"},
              {"value", initializer}}
      }
  };
  EXPECT_EQ(var->toJson(), j);
}