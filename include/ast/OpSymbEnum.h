#ifndef AST_OPTIMIZER_INCLUDE_AST_OPSYMBENUM_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPSYMBENUM_H_

#include <variant>
#include <string>
#include "LiteralInt.h"
#include "LiteralBool.h"

enum ArithmeticOp : char {
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

// generate a typedef for this std::variant to ensure that always the same Enums order is used
typedef std::variant<ArithmeticOp, LogCompOp, UnaryOp> OpSymbolVariant;

class OpSymb {
 public:
  static std::string getTextRepr(ArithmeticOp bop) {
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

  static std::string getTextRepr(OpSymbolVariant opVar) {
    switch (opVar.index()) {
      case 0:return getTextRepr(std::get<ArithmeticOp>(opVar));
      case 1:return getTextRepr(std::get<LogCompOp>(opVar));
      case 2:return getTextRepr(std::get<UnaryOp>(opVar));
      default:return "";
    }
  }

  static AbstractLiteral *getIdentityElement(OpSymbolVariant op) {
    if (op.index()==0) {  // op is ArithmeticOp
      auto arithmeticOp = std::get<ArithmeticOp>(op);
      // identity element only exists for add, sub, mult
      static const std::vector<int> identityElements = {0, 0, 1};
      if (arithmeticOp > identityElements.size())
        throw std::logic_error("Identity element not defined for given operator: " + getTextRepr(arithmeticOp));
      return new LiteralInt(identityElements.at(arithmeticOp));

    } else if (op.index()==1) {  // op is LogCompOp
      auto logCompOp = std::get<LogCompOp>(op);
      // identity element only exists for AND, OR, XOR
      static const std::vector<bool> identityElements = {true, false, false};
      if (logCompOp > identityElements.size())
        throw std::logic_error("Identity element not defined for given operator: " + getTextRepr(logCompOp));
      return new LiteralBool(identityElements.at(logCompOp));

    } else if (op.index()==2) {  // op is UnaryOp
      throw std::logic_error("Identity element for unary operator unsupported! Supported for binary operators only.");

    } else {
      throw std::logic_error("Unknown operator given, cannot determine identity element!");
    }
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPSYMBENUM_H_
