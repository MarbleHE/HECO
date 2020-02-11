#include <iostream>
#include <Function.h>
#include <Operator.h>
#include <If.h>
#include <LiteralInt.h>
#include <LiteralBool.h>
#include <LiteralFloat.h>
#include <LogicalExpr.h>
#include <VarAssignm.h>
#include <Block.h>
#include <Return.h>
#include <BinaryExpr.h>
#include <Ast.h>
#include <CallExternal.h>
#include <While.h>
#include <UnaryExpr.h>
#include <Call.h>
#include <Group.h>

#include "../../include/visitor/PrintVisitor.h"
#include "../../main.h"
#include "../utilities/Scope.h"

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

void PrintVisitor::visit(BinaryExpr &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Block &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Call &elem) {
  Node* node = static_cast<Node*>(static_cast<AbstractStatement*>(&elem));
  addOutputStr(*node);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(CallExternal &elem) {
  Node* node = static_cast<Node*>(static_cast<AbstractStatement*>(&elem));
  addOutputStr(*node, {elem.getFunctionName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Function &elem) {
  addOutputStr(elem, {elem.getName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(FunctionParameter &elem) {
  addOutputStr(elem, {elem.getDatatype()->toString()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Group &elem) {
  addOutputStr(elem);
  // group statements
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(If &elem) {
  addOutputStr(elem);
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(LiteralBool &elem) {
  addOutputStr(elem, {elem.getTextValue()});
}

void PrintVisitor::visit(LiteralInt &elem) {
  addOutputStr(elem, {std::to_string(elem.getValue())});
}

void PrintVisitor::visit(LiteralString &elem) {
  addOutputStr(elem, {elem.getValue()});
}

void PrintVisitor::visit(LiteralFloat &elem) {
  addOutputStr(elem, {std::to_string(elem.getValue())});
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
  addOutputStr(elem, {elem.getDatatype()->toString() + " " + elem.getIdentifier()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Variable &elem) {
  addOutputStr(elem, {elem.getIdentifier()});
}

void PrintVisitor::visit(While &elem) {
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

void PrintVisitor::printNodeName(Node &node) {
  ss << getIndentation() << node.getNodeName() << ":";
}

void PrintVisitor::addOutputStr(Node &node) {
  printNodeName(node);
  printScope();
}

void PrintVisitor::addOutputStr(Node &node, const std::list<std::string> &args) {
  printNodeName(node);
  // print AST node type (e.g., FunctionParameter)
  ss << " ";
  // print primitive parameters related to AST (e.g., int, string)
  for (auto &arg : args) ss << "(" << arg << ")";
  printScope();
}

void PrintVisitor::printScope() {
  // only print scope where statement belongs to if statement has changed since last print
  if (getLastPrintedScope() == nullptr || curScope != getLastPrintedScope()) {
    ss << "\t[";
    ss << this->curScope->getScopeIdentifier();
    ss << "]";
    setLastPrintedScope(curScope);
  }
  ss << std::endl;
}

Scope* PrintVisitor::getLastPrintedScope() const {
  return lastPrintedScope;
}

void PrintVisitor::setLastPrintedScope(Scope* scope) {
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
