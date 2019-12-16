
#include <iostream>
#include <Function.h>
#include <Operator.h>
#include <If.h>
#include <LiteralInt.h>
#include <LiteralBool.h>
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
#include <Class.h>
#include <sstream>

#include "../../include/visitor/PrintVisitor.h"
#include "../../main.h"


void PrintVisitor::visit(Ast &elem) {
    _DEBUG_RUNNING();
    Visitor::visit(elem);
}

void PrintVisitor::visit(BinaryExpr &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(Block &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(Call &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(CallExternal &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getFunctionName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(Class &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getName(), elem.getSuperclass()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(Function &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(FunctionParameter &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getDatatype() + " " + elem.getIdentifier()});
}

void PrintVisitor::visit(Group &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    // group statements
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(If &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(LiteralBool &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getTextValue()});
}

void PrintVisitor::visit(LiteralInt &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), std::to_string(elem.getValue())});
}

void PrintVisitor::visit(LiteralString &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getValue()});
}

void PrintVisitor::visit(LogicalExpr &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(Operator &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getOperatorString()});
}

void PrintVisitor::visit(Return &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(UnaryExpr &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(VarAssignm &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getIdentifier()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(VarDecl &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getDatatype() + " " + elem.getIdentifier()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();
}

void PrintVisitor::visit(Variable &elem) {
    std::cout << formatOutputStr({elem.getNodeName(), elem.getIdentifier()});
}

void PrintVisitor::visit(While &elem) {
    std::cout << formatOutputStr({elem.getNodeName()});
    this->incrementLevel();
    Visitor::visit(elem);
    this->decrementLevel();


}

// ------------–------
// Constructor & Utility functions
// ------------–------

PrintVisitor::PrintVisitor(const int level) : level(level) {}

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

std::string PrintVisitor::formatOutputStr(const std::list<std::string> &args) {
    std::ostringstream ss;
    // print AST node type (e.g., FunctionParameter)
    ss << getIndentation() << args.front() << ": ";
    // print primitive parameters related to AST (e.g., int, string)
    for (auto it = std::next(args.begin()); it != args.end(); ++it) {
        ss << "(" << *it << ")";
    }
    ss << std::endl;
    return ss.str();
}
