#ifndef AST_OPTIMIZER_INCLUDE_AST_OPSYMBENUM_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPSYMBENUM_H_

#include <variant>
#include <string>

class OpSymb {
 public:
  enum BinaryOp : char {
    // arithmetic operator
    addition = 0, subtraction, multiplication, division, modulo,
  };

  enum LogCompOp : char {
    // logical operator
    logicalAnd = 0, logicalOr, logicalXor,
    // relational operator
    smaller, smallerEqual, greater, greaterEqual, equal, unequal
  };

  enum UnaryOp : char {
    // logical operator
    negation = 0
  };

  static std::string getTextRepr(BinaryOp bop) {
    static const std::string binaryOpStrings[] = {"add", "sub", "mult", "div", "mod"};
    return binaryOpStrings[bop];
  }

  static std::string getTextRepr(LogCompOp lcop) {
    static const std::string logicalOpStrings[] = {"AND", "OR", "XOR", "<", "<=", ">", ">=", "==", "!="};
    return logicalOpStrings[lcop];
  }

  static std::string getTextRepr(UnaryOp uop) {
    static const std::string unaryOpStrings[] = {"!"};
    return unaryOpStrings[uop];
  }

  static std::string getTextRepr(std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> opVar) {
    switch (opVar.index()) {
      case 0:return getTextRepr(std::get<OpSymb::BinaryOp>(opVar));
      case 1:return getTextRepr(std::get<OpSymb::LogCompOp>(opVar));
      case 2:return getTextRepr(std::get<OpSymb::UnaryOp>(opVar));
      default:return "";
    }
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPSYMBENUM_H_
