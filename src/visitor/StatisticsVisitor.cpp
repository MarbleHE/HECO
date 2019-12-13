
#include <iostream>
#include <Function.h>
#include <Operator.h>
#include "../../include/visitor/StatisticsVisitor.h"

void StatisticsVisitor::visit(Ast &elem) {
    std::cout << "object Ast visited." << std::endl;

}

void StatisticsVisitor::visit(BinaryExpr &elem) {
    std::cout << "object BinaryExpr visited." << std::endl;

}

void StatisticsVisitor::visit(Block &elem) {
    std::cout << "object Block visited." << std::endl;
}

void StatisticsVisitor::visit(Call &elem) {
    std::cout << "object Call visited." << std::endl;;
}

void StatisticsVisitor::visit(CallExternal &elem) {
    std::cout << "object CallExternal visited." << std::endl;

}

void StatisticsVisitor::visit(Class &elem) {
    std::cout << "object Class visited." << std::endl;

}

void StatisticsVisitor::visit(Function &elem) {
    std::cout << "object Function visited." << std::endl;
    std::cout << elem.getName() << std::endl;

}

void StatisticsVisitor::visit(FunctionParameter &elem) {
    std::cout << "object FunctionParameter visited." << std::endl;

}

void StatisticsVisitor::visit(Group &elem) {
    std::cout << "object Group visited." << std::endl;

}

void StatisticsVisitor::visit(If &elem) {
    std::cout << "object If visited." << std::endl;

}

void StatisticsVisitor::visit(Literal &elem) {
    std::cout << "object Literal visited." << std::endl;

}

void StatisticsVisitor::visit(LiteralBool &elem) {
    std::cout << "object LiteralBool visited." << std::endl;

}

void StatisticsVisitor::visit(LiteralInt &elem) {
    std::cout << "object LiteralInt visited." << std::endl;

}

void StatisticsVisitor::visit(LiteralString &elem) {
    std::cout << "object LiteralString visited." << std::endl;

}

void StatisticsVisitor::visit(LogicalExpr &elem) {
    std::cout << "object LogicalExpr visited." << std::endl;

}

void StatisticsVisitor::visit(Operator &elem) {
    std::cout << "object Operator visited." << std::endl;

}

void StatisticsVisitor::visit(Return &elem) {
    std::cout << "object Return visited." << std::endl;

}

void StatisticsVisitor::visit(UnaryExpr &elem) {
    std::cout << "object UnaryExpr visited." << std::endl;

}

void StatisticsVisitor::visit(VarAssignm &elem) {
    std::cout << "object VarAssignm visited." << std::endl;

}

void StatisticsVisitor::visit(VarDecl &elem) {
    std::cout << "object VarDecl visited." << std::endl;

}

void StatisticsVisitor::visit(Variable &elem) {
    std::cout << "object Variable visited." << std::endl;

}

void StatisticsVisitor::visit(While &elem) {
    std::cout << "object While visited." << std::endl;

}