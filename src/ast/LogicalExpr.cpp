#include "../../include/ast/LogicalExpr.h"
#include <iostream>
#include <LiteralInt.h>
#include <Variable.h>
#include <LiteralBool.h>

json LogicalExpr::toJson() const {
    json j;
    j["type"] = "LogicalExpr";
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op;
    j["rightOperand"] = this->right->toJson();
    return j;
}


LogicalExpr::LogicalExpr(AbstractExpr *left, LogicalCompOperator op, AbstractExpr *right) :
        left(left), op(op), right(right) {}

//template <typename T>
//LogicalExpr::LogicalExpr(T variableLeft, LogicalCompOperator op, T variableRight) {
//    this->op = op;
//    if (std::is_same<T, int>::value) {
//        left = new LiteralInt(variableLeft);
//        right = new LiteralInt(variableRight);
//    } else if (std::is_same<T, std::string>::value){
//        left = new Variable(variableLeft);
//        right = new Variable(variableRight);
//    } else if (std::is_same<T, bool>::value){
//        left = new LiteralBool(variableLeft);
//        right = new LiteralBool(variableRight);
//    }
//}
