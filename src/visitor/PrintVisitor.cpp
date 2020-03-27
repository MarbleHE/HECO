#include <iostream>
#include "Function.h"
#include "Operator.h"
#include "If.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralFloat.h"
#include "LogicalExpr.h"
#include "VarAssignm.h"
#include "Block.h"
#include "Return.h"
#include "ArithmeticExpr.h"
#include "Ast.h"
#include "CallExternal.h"
#include "While.h"
#include "UnaryExpr.h"
#include "Call.h"
#include "PrintVisitor.h"
#include "Scope.h"
#include "For.h"
#include "Rotate.h"
#include "Transpose.h"
#include "GetMatrixElement.h"
#include "OperatorExpr.h"

template<typename T>
void PrintVisitor::printChildNodesIndented(T &elem) {
  incrementLevel();  // increment the indentation (tabs) used by addOutputStr
  Visitor::visit(elem); // continue traversal by visiting child nodes
  decrementLevel();
}

void PrintVisitor::visit(Ast &elem) {
  resetVisitor();
  Visitor::visit(elem);
  // at the end of traversal print the tree if printScreen is true
  if (printScreen) std::cout << ss.str() << std::endl;
}

void PrintVisitor::visit(ArithmeticExpr &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Block &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Call &elem) {
  auto *node = static_cast<AbstractNode *>(static_cast<AbstractExpr *>(&elem));
  addOutputStr(*node);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(CallExternal &elem) {
  addOutputStr(elem, {elem.getFunctionName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Function &elem) {
  addOutputStr(elem, {elem.getName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(FunctionParameter &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(If &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(GetMatrixElement &elem) {
  printMatrixIndex();
  this->decrementLevel();
  addOutputStr(elem);
  this->incrementLevel();
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(LiteralBool &elem) {
  if (elem.getMatrix()->containsAbstractExprs()) {
    addOutputStr(elem);
    printChildNodesIndented(elem);
  } else {
    addOutputStr(elem, {elem.getMatrix()->toString()});
  }
}

void PrintVisitor::visit(LiteralInt &elem) {
  if (elem.getMatrix()->containsAbstractExprs()) {
    addOutputStr(elem);
    printChildNodesIndented(elem);
  } else {
    addOutputStr(elem, {elem.getMatrix()->toString()});
  }
}

void PrintVisitor::visit(LiteralString &elem) {
  if (elem.getMatrix()->containsAbstractExprs()) {
    addOutputStr(elem);
    printChildNodesIndented(elem);
  } else {
    addOutputStr(elem, {elem.getMatrix()->toString()});
  }
}

void PrintVisitor::visit(LiteralFloat &elem) {
  if (elem.getMatrix()->containsAbstractExprs()) {
    addOutputStr(elem);
    printChildNodesIndented(elem);
  } else {
    addOutputStr(elem, {elem.getMatrix()->toString()});
  }
}

void PrintVisitor::visit(LogicalExpr &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Operator &elem) {
  addOutputStr(elem, {elem.getOperatorString()});
}

void PrintVisitor::visit(Return &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(UnaryExpr &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(VarAssignm &elem) {
  addOutputStr(elem, {elem.getIdentifier()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(VarDecl &elem) {
  addOutputStr(elem, {elem.getIdentifier()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Variable &elem) {
  addOutputStr(elem, {elem.getIdentifier()});
}

void PrintVisitor::visit(While &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(For &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(ParameterList &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Rotate &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Transpose &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(OperatorExpr &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

// ------------–------
// Constructor & Utility functions
// ------------–------

PrintVisitor::PrintVisitor() : PrintVisitor(true) {
}

PrintVisitor::PrintVisitor(bool printScreen) : printScreen(printScreen), level(0), lastPrintedScope(nullptr) {}

void PrintVisitor::incrementLevel() {
  this->level = (this->level + 1);
}

void PrintVisitor::decrementLevel() {
  this->level = (this->level - 1);
}

std::string PrintVisitor::getIndentation() {
  return std::string(this->level, '\t');
}

void PrintVisitor::resetLevel() {
  this->level = 0;
}

void PrintVisitor::printNodeName(AbstractNode &node) {
  if (showUniqueNodeIds) {
    ss << getIndentation() << node.getUniqueNodeId() << ":";
  } else {
    ss << getIndentation() << node.getNodeType() << ":";
  }
}

void PrintVisitor::addOutputStr(AbstractNode &node) {
  printNodeName(node);
  printScope();
}

void PrintVisitor::addOutputStr(AbstractNode &node, const std::list<std::string> &args) {
  printNodeName(node);
  // print AST node type (e.g., FunctionParameter)
  ss << " ";
  // print primitive parameters related to AST (e.g., int, string)
  for (auto &arg : args) ss << "(" << arg << ")";
  printScope();
}

void PrintVisitor::printScope() {
  // only print scope where statement belongs to if statement has changed since last print
  if (getLastPrintedScope()==nullptr || curScope!=getLastPrintedScope()) {
    ss << "\t[";
    ss << this->curScope->getScopeIdentifier();
    ss << "]";
    setLastPrintedScope(curScope);
  }
  ss << std::endl;
}

Scope *PrintVisitor::getLastPrintedScope() const {
  return lastPrintedScope;
}

void PrintVisitor::setLastPrintedScope(Scope *scope) {
  this->lastPrintedScope = scope;
}

PrintVisitor::~PrintVisitor() {
  delete lastPrintedScope;
}

std::string PrintVisitor::getOutput() const {
  return ss.str();
}

void PrintVisitor::resetVisitor() {
  resetLevel();
  lastPrintedScope = nullptr;
  ss.clear();
  ss.str(std::string());
}

void PrintVisitor::useUniqueNodeIds(bool value) {
  this->showUniqueNodeIds = value;
}

void PrintVisitor::visit(Datatype &elem) {
  std::string encryption = (elem.isEncrypted() ? "encrypted" : "plaintext");
  addOutputStr(elem, {encryption + " " + Datatype::enumToString(elem.getType())});
  Visitor::visit(elem);
}

void PrintVisitor::printMatrixIndex() {
  ss << "(" << nextMatrixIndexToBePrinted.first << "," << nextMatrixIndexToBePrinted.second << ")";
}
