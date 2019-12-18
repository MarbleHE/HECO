#ifndef MASTER_THESIS_CODE_OPERATOR_H
#define MASTER_THESIS_CODE_OPERATOR_H

#include "../visitor/Visitor.h"
#include "Node.h"

class OpSymb {
protected:
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
                negation = 0,
        // arithmetic operator
                increment, decrement
    };

    static std::string getTextRepr(BinaryOp bop) {
        static const std::string binaryOpStrings[] = {"add", "sub", "mult", "div", "mod"};
        return binaryOpStrings[bop];
    }

    static std::string getTextRepr(LogCompOp lcop) {
        static const std::string logicalOpStrings[] = {"AND", "OR", "XOR", "<", "<=", ">", ">=", "!="};
        return logicalOpStrings[lcop];
    }

    static std::string getTextRepr(UnaryOp uop) {
        static const std::string unaryOpStrings[] = {"!", "++", "--"};
        return unaryOpStrings[uop];
    }
};


class Operator : public Node {
private:
    std::string operatorString;
public:
    explicit Operator(OpSymb::LogCompOp op);

    explicit Operator(OpSymb::BinaryOp op);

    explicit Operator(OpSymb::UnaryOp op);

    [[nodiscard]] const std::string &getOperatorString() const;

    virtual void accept(Visitor &v);

    [[nodiscard]] std::string getNodeName() const override;

};


#endif //MASTER_THESIS_CODE_OPERATOR_H
