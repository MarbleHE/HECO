#include "ast_opt/utilities/Operator.h"

std::string toString(ArithmeticOp bop) {
  static const std::string binaryOpStrings[] = {"add", "sub", "mult", "div", "mod"};
  return binaryOpStrings[bop];
}

std::string toString(LogicalOp logop) {
  static const std::string logicalOpStrings[] = {"AND", "OR", "XOR", "<", "<=", ">", ">=", "==", "!="};
  return logicalOpStrings[logop];
}

std::string toString(UnaryOp uop) {
  static const std::string unaryOpStrings[] = {"!"};
  return unaryOpStrings[uop];
}

std::string toString(OperatorVariant opVar) {
  switch (opVar.index()) {
    case 0:return toString(std::get<ArithmeticOp>(opVar));
    case 1:return toString(std::get<LogicalOp>(opVar));
    case 2:return toString(std::get<UnaryOp>(opVar));
    default:return "";
  }
}

Operator::Operator(OperatorVariant op)  : op(op) {}

std::string Operator::toString() const {
  return ::toString(op);
}
