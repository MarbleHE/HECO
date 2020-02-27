#include <iostream>
#include "MultRewriteVisitor.h"
#include "ArithmeticExpr.h"
#include "Block.h"
#include "Variable.h"

void MultRewriteVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(ArithmeticExpr &elem) {
  // If current ArithmeticExpr is a multiplication
  if (elem.getOp()->equals(OpSymb::ArithmeticOp::multiplication)) {
    // A. For case "int result = (A * (B * C))" where multiple ArithmeticExpr are in the same statement
    if (auto lStat = curScope->getLastStatement()) {
      // If the statement contains another (higher tree level) ArithmeticExpr (exclude subtree of cur. ArithmeticExpr) ...
      if (ArithmeticExpr *lastStat = lStat->contains(new ArithmeticExpr(OpSymb::multiplication), &elem)) {
        // ... then swap previousAexpLeftOp with currentAexpRightOp
        ArithmeticExpr::swapOperandsLeftAWithRightB(lastStat, &elem);
        numChanges++;
      }
    }

    // B. For case { int tmp = B*C; tmp = tmp*A; } where both BinaryExp are in separate statements.
    // If there is a last statement (i.e., this is not the first statement of the scope).
    // (-> Check penultimate statement b/c the statement this ArithmeticExpr elem belongs to was already added to curScope)
    if (auto puStat = curScope->getNthLastStatement(2)) {
      // If previous statement in scope contains a ArithmeticExpr multiplication...
      if (ArithmeticExpr *lastStat = puStat->contains(
          new ArithmeticExpr(OpSymb::multiplication), nullptr)) {
        // Retrieve variable identifier from last statement (VarDecl or VarAssignm)
        std::string puVarTargetIdentifier = puStat->getVarTargetIdentifier();
        std::string curAexpTargetIdentifier = curScope->getLastStatement()->getVarTargetIdentifier();
        // Check that left operand reuses variable of previous statement in its left branch
        if (!puVarTargetIdentifier.empty() && elem.getLeft()->contains(new Variable(puVarTargetIdentifier))) {
          // Check that both target variables are the same, otherwise this transformation will change semantics.
          // For example, here rewriting is NOT applicable: { int tmp = B*C; int tmp2 = tmp*A; }
          if (puVarTargetIdentifier==curAexpTargetIdentifier) {
            // ... then swap previousAexpLeftOp with currentAexpRightOp
            ArithmeticExpr::swapOperandsLeftAWithRightB(lastStat, &elem);
            numChanges++;
          }
        }
      }
    }
  } // end: if this ArithmeticExpr is a multiplication
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Block &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Call &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(CallExternal &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Function &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(FunctionParameter &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(If &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LiteralBool &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LiteralInt &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LiteralFloat &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LiteralString &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LogicalExpr &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Operator &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Return &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(VarAssignm &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(VarDecl &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Variable &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(While &elem) {
  Visitor::visit(elem);
}

int MultRewriteVisitor::getNumChanges() const {
  return numChanges;
}

bool MultRewriteVisitor::changedAst() const {
  return numChanges!=0;
}


