#include "ast_opt/ast_utilities/Visitor.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/Literal.h"
#include "gtest/gtest.h"

/// Minimal example of a "SpecialVisitor" to be used with the visitor template
class SpecialVisitor : public ScopedVisitor {
 public:

  /// Helper to figure out what was called
  std::string message = "";

  /// Non-inheriting visit(...) function hides Base class visit(...) for all other types

  /// Base Class function, should be called e.g. for LiteralInt, etc
  void visit(AbstractExpression &) {
    message = "SpecialVisitor::visit(AbstractExpression&)";
  }

  /// Specific function that specifies functionality for one derived class
  void visit(LiteralBool &) {
    message = "SpecialVisitor::visit(LiteralBool&)";
  }

};

TEST(VisitorTemplate, has_visit) {
  // This test confirms that the has_visit SFINAE detection works as expected

  // Parentheses required to avoid breaking the GTEST macro
  EXPECT_EQ((has_visit<SpecialVisitor, AbstractExpression &>), true);
  EXPECT_EQ((has_visit<SpecialVisitor, LiteralBool &>), true);
  EXPECT_EQ((has_visit<SpecialVisitor, LiteralInt &>), true); // because of LiteralInt : public AbstractExpression
  EXPECT_EQ((has_visit<SpecialVisitor, AbstractNode &>), false);
  EXPECT_EQ((has_visit<SpecialVisitor, AbstractStatement &>), false);
  EXPECT_EQ((has_visit<SpecialVisitor, Assignment &>), false);
}

TEST(VisitorTemplate, dispatchDefault) {
  // When calling visit on a type T where neither T nor any base class of T
  // has a visit(...) function in SpecialVisitor, the scoped visitor should be called
  // As a consequence, the message string should remain empty.

  auto r = Return();

  Visitor<SpecialVisitor> v;
  v.visit(r);

  EXPECT_EQ(v.message, "");
}

TEST(VisitorTemplate, dispatchBaseClass) {
  // When calling visit on a type T where SpecialVisitor::visit(T&) does not exist,
  // but there exists SpecialVisitor::visit(B&) with B a base class of T
  // Then this base class function should be called

  LiteralInt i(42);

  // Confirm that SpecialVisitor can be called directly on this
  SpecialVisitor sv;
  sv.visit(i);

  // Confirm that Visitor<SpecialVisitor> routes to the same
  Visitor<SpecialVisitor> v;
  v.visit(i);

  EXPECT_EQ(sv.message, "SpecialVisitor::visit(AbstractExpression&)");
  EXPECT_EQ(v.message, "SpecialVisitor::visit(AbstractExpression&)");
}

TEST(VisitorTemplate, dispatchDerivedClass) {
  // When calling visit on a type T where SpecialVisitor::visit(T&) does  exist,
  // this derived  class function should be called, even if there exists SpecialVisitor::visit(B&) with B a base class of T


  LiteralBool b(true);
  // Confirm that SpecialVisitor can be called directly on this
  SpecialVisitor sv;
  b.accept(sv);

  // Confirm that Visitor<SpecialVisitor> routes to the same
  Visitor<SpecialVisitor> v;
  b.accept(v);

  EXPECT_EQ(sv.message, "SpecialVisitor::visit(LiteralBool&)");
  EXPECT_EQ(v.message, "SpecialVisitor::visit(LiteralBool&)");
}