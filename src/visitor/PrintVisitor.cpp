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
  if (printScreen) {
    std::cout << ss.str() << std::endl;
  }
}

void PrintVisitor::visit(BinaryExpr &elem) {
  addOutputStr({elem.getNodeName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Block &elem) {
  addOutputStr({elem.getNodeName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Call &elem) {
  addOutputStr({elem.getNodeName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(CallExternal &elem) {
  addOutputStr({elem.getNodeName(), elem.getFunctionName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Function &elem) {
  addOutputStr({elem.getNodeName(), elem.getName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(FunctionParameter &elem) {
  addOutputStr({elem.getNodeName(), elem.getDatatype()->toString()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Group &elem) {
  addOutputStr({elem.getNodeName()});
  // group statements
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(If &elem) {
  addOutputStr({elem.getNodeName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(LiteralBool &elem) {
  addOutputStr({elem.getNodeName(), elem.getTextValue()});
}

void PrintVisitor::visit(LiteralInt &elem) {
  addOutputStr({elem.getNodeName(), std::to_string(elem.getValue())});
}

void PrintVisitor::visit(LiteralString &elem) {
  addOutputStr({elem.getNodeName(), elem.getValue()});
}

void PrintVisitor::visit(LiteralFloat &elem) {
  addOutputStr({elem.getNodeName(), std::to_string(elem.getValue())});
}

void PrintVisitor::visit(LogicalExpr &elem) {
  addOutputStr({elem.getNodeName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Operator &elem) {
  addOutputStr({elem.getNodeName(), elem.getOperatorString()});
}

void PrintVisitor::visit(Return &elem) {
  addOutputStr({elem.getNodeName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(UnaryExpr &elem) {
  addOutputStr({elem.getNodeName()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(VarAssignm &elem) {
  addOutputStr({elem.getNodeName(), elem.getIdentifier()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(VarDecl &elem) {
  addOutputStr({elem.getNodeName(), elem.getDatatype()->toString() + " " + elem.getIdentifier()});
  printChildNodesIndented(elem);
}

void PrintVisitor::visit(Variable &elem) {
  addOutputStr({elem.getNodeName(), elem.getIdentifier()});
}

void PrintVisitor::visit(While &elem) {
  addOutputStr({elem.getNodeName()});
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

void PrintVisitor::addOutputStr(const std::list<std::string> &args) {
  // TODO(pjattke) refactor elem into addOutputStr to enforce NodeName in output
  //std::ostringstream tempStream;
  // print AST node type (e.g., FunctionParameter)
  ss << getIndentation() << args.front() << ":";
  if (args.size() > 1) ss << " ";
  // print primitive parameters related to AST (e.g., int, string)
  for (auto it = std::next(args.begin()); it != args.end(); ++it) {
    ss << "(" << *it << ")";
  }
  // only print scope where statement belongs to if statement has changed since last print
  if (getLastPrintedScope() == nullptr || curScope != getLastPrintedScope()) {
    ss << "\t[" << this->curScope->getScopeIdentifier() << "]";
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
