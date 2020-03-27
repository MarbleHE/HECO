#include <iostream>
#include "MultRewriteVisitor.h"
#include "ArithmeticExpr.h"
#include "OperatorExpr.h"
#include "Block.h"
#include "Variable.h"

void MultRewriteVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

void MultRewriteVisitor::visit(ArithmeticExpr &elem) {
  // If current ArithmeticExpr is a multiplication
  if (elem.getOperator()->equals(ArithmeticOp::MULTIPLICATION)) {
    // A. For case "int result = (A * (B * C))" where multiple ArithmeticExpr are in the same statement
    if (auto lStat = curScope->getLastStatement()) {
      // If statement contains another (higher tree level) ArithmeticExpr (exclude subtree of cur. ArithmeticExpr) ...
      if (auto lastStat = lStat->contains(new ArithmeticExpr(ArithmeticOp::MULTIPLICATION), &elem)) {
        // ... then swap previousAexpLeftOp with currentAexpRightOp
        ArithmeticExpr::swapOperandsLeftAWithRightB(lastStat, &elem);
        numChanges++;
      }
    }

    // B. For case { int tmp = B*C; tmp = tmp*A; } where both BinaryExp are in separate statements.
    // If there is a last statement (i.e., this is not the first statement of the scope). Check penultimate statement
    // because the statement this ArithmeticExpr elem belongs to was already added to curScope.
    if (auto puStat = curScope->getNthLastStatement(2)) {
      // If previous statement in scope contains a ArithmeticExpr multiplication...
      if (auto lastStat = puStat->contains(new ArithmeticExpr(ArithmeticOp::MULTIPLICATION), nullptr)) {
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

void MultRewriteVisitor::visit(OperatorExpr &elem) {
  throw std::runtime_error("Unimplemented: OperatorExpr not supported by MultRewriteVisitor!");
}

int MultRewriteVisitor::getNumChanges() const {
  return numChanges;
}

bool MultRewriteVisitor::changedAst() const {
  return numChanges!=0;
}
