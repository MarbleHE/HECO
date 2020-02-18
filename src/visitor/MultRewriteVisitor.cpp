#include <iostream>
#include "MultRewriteVisitor.h"
#include "BinaryExpr.h"
#include "Block.h"
#include "Variable.h"

void MultRewriteVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(BinaryExpr &elem) {
  // If current BinaryExpr is a multiplication
  if (elem.getOp()->equals(OpSymb::BinaryOp::multiplication)) {
    // A. For case "int result = (A * (B * C))" where multiple BinaryExpr are in the same statement
    if (auto lStat = curScope->getLastStatement()) {
      // If the statement contains another (higher tree level) BinaryExpr (exclude subtree of cur. BinaryExpr) ...
      if (BinaryExpr *lastStat = lStat->contains(new BinaryExpr(OpSymb::multiplication), &elem)) {
        // ... then swap previousBexpLeftOp with currentBexpRightOp
        BinaryExpr::swapOperandsLeftAWithRightB(lastStat, &elem);
        numChanges++;
      }
    }

    // B. For case { int tmp = B*C; tmp = tmp*A; } where both BinaryExp are in separate statements.
    // If there is a last statement (i.e., this is not the first statement of the scope).
    // (-> Check penultimate statement b/c the statement this BinaryExpr elem belongs to was already added to curScope)
    if (auto puStat = curScope->getNthLastStatement(2)) {
      // If previous statement in scope contains a BinaryExpr multiplication...
      if (BinaryExpr *lastStat = puStat->contains(
          new BinaryExpr(OpSymb::multiplication), nullptr)) {
        // Retrieve variable identifier from last statement (VarDecl or VarAssignm)
        std::string puVarTargetIdentifier = puStat->getVarTargetIdentifier();
        std::string curBexpTargetIdentifier = curScope->getLastStatement()->getVarTargetIdentifier();
        // Check that left operand reuses variable of previous statement in its left branch
        if (!puVarTargetIdentifier.empty() && elem.getLeft()->contains(new Variable(puVarTargetIdentifier))) {
          // Check that both target variables are the same, otherwise this transformation will change semantics.
          // For example, here rewriting is NOT applicable: { int tmp = B*C; int tmp2 = tmp*A; }
          if (puVarTargetIdentifier==curBexpTargetIdentifier) {
            // ... then swap previousBexpLeftOp with currentBexpRightOp
            BinaryExpr::swapOperandsLeftAWithRightB(lastStat, &elem);
            numChanges++;
          }
        }
      }
    }
  } // end: if this BinaryExpr is a multiplication
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


