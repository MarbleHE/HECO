#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VISITOR_H_

#include "ast_opt/visitor/IVisitor.h"

/// This class implements the "default" behaviour of a visitor
/// simply visiting a node's children
/// and setting the scope as required
class ScopedVisitor : public IVisitor {
 public:
  void visit(BinaryExpression &elem) override;

  void visit(Block &elem) override;

  void visit(For &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(If &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralChar &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LiteralDouble &elem) override;

  void visit(LiteralString &elem) override;

  void visit(UnaryExpression &elem) override;

  void visit(VariableAssignment &elem) override;

  void visit(VariableDeclaration &elem) override;

  void visit(Variable &elem) override;
};


/// SFINAE based detection if T::visit(Args...) exists
/// Taken from https://stackoverflow.com/a/28309612
template<typename T, typename... Args>
class is_visit_available {
  template<typename C,
      typename = decltype(std::declval<C>().visit(std::declval<Args>()...))>
  static std::true_type test(int);

  template<typename C>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/// Does T::visit(Args...) exist? (Syntactic sugar for is_visit_available)
template<typename T, typename... Args>
constexpr bool has_visit = is_visit_available<T, Args...>::value;

/// Helper Class to create Visitors.
/// This allows a SpecialVisitor to not only provide only *some* of the visit functions,
/// with the rest being handled by the default ScopedVisitor::visit() function
/// but it also allows the SpecialVisitor to target Superclasses, e.g. visit(AbstractExpression)
///
/// SpecialVisitor should inherit ScopedVisitor publicly,
/// should define at least one visit(Some AST Type) WITHOUT specifying override
/// (intentionally hiding all base class visit(...) overloads)
/// and should not include "using IVisitor::visit" or "using ScopedVisitor::visit" (which would un-hide them again)
///
/// This leads to only the defined visit(..) functions being accessible in SpecialVisitor
/// The Visitor<..> template than redirects any calls for which SpecialVisitor does not have a visit overload
/// to the default visit(elem) method in ScopedVisitor.
/// In addition, you can define visit(BaseClass &elem) in SpecialVisitor and,
/// unless there is a more specific visit(DerivedClass &elem) with DerivedClass : public BaseClass
/// All calls for classes derived from BaseClass will be handled by this visit.
/// This is not generally possible with normal visitors because they must have all possible visit functions
/// and therefore always have a more specific one for the derived classes.
///
/// SpecialVisitor should also specify any desired constructors, Visitor<SpecialVisitor> will inherit and expose them
///
/// Note that IVisitor, ScopedVistor and the Visitor<..> template only need functions
/// corresponding to the concrete classes in the Node hierarchy.
/// Functions applying to parent classes like AbstractExpression need to appear only in the SpecialVisitor
/// The way this works is that Visitor<..>'s visit function for the concrete class will be called,
/// but if SpecialVisitor offers a function for the base class but not the concrete derived class,
/// the function for the (potentially abstract) base class will be called by Visitor<...>'s visit(concrete&)
///
/// \tparam SpecialVisitor  The class implementing the actual visiting logic
template <typename SpecialVisitor>
class Visitor : public SpecialVisitor{
 public:
  /// Ensure that SpecialVisitor is actually a visitor
  static_assert(std::is_base_of<IVisitor,SpecialVisitor>::value);

  /// Inherit Constructors from SpecialVisitor
  using SpecialVisitor::SpecialVisitor;

  void visit(BinaryExpression &elem) override {
    if constexpr (has_visit<SpecialVisitor, BinaryExpression&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }

  }

  void visit(Block &elem) override {
    if constexpr (has_visit<SpecialVisitor,Block&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(For &elem) override {
    if constexpr (has_visit<SpecialVisitor,For&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(If &elem) override {
    if constexpr (has_visit<SpecialVisitor,If&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(LiteralBool &elem) override {
    if constexpr (has_visit<SpecialVisitor,LiteralBool&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(LiteralChar &elem) override {
    if constexpr (has_visit<SpecialVisitor,LiteralChar&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(LiteralInt &elem) override {
    if constexpr (has_visit<SpecialVisitor,LiteralInt&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(LiteralFloat &elem) override {
    if constexpr (has_visit<SpecialVisitor,LiteralFloat&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(LiteralDouble &elem) override {
    if constexpr (has_visit<SpecialVisitor,LiteralDouble&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(LiteralString &elem) override {
    if constexpr (has_visit<SpecialVisitor,LiteralString&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(UnaryExpression &elem) override {
    if constexpr (has_visit<SpecialVisitor, UnaryExpression&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(VariableAssignment &elem) override {
    if constexpr (has_visit<SpecialVisitor,VariableAssignment&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(VariableDeclaration &elem) override {
    if constexpr (has_visit<SpecialVisitor, VariableDeclaration&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }

  void visit(Variable &elem) override {
    if constexpr (has_visit<SpecialVisitor,Variable&>)  {
      this->SpecialVisitor::visit(elem);
    } else {
      this->ScopedVisitor::visit(elem);
    }
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VISITOR_H_
