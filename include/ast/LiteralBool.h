#ifndef AST_OPTIMIZER_INCLUDE_LITERALBOOL_H
#define AST_OPTIMIZER_INCLUDE_LITERALBOOL_H

#include "Literal.h"
#include <string>
#include <map>

class LiteralBool : public Literal {
private:
    bool value;

    Node *createClonedNode(bool keepOriginalUniqueNodeId) override;

public:
    explicit LiteralBool(bool value);

    ~LiteralBool() override;

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] bool getValue() const;

    [[nodiscard]] std::string getTextValue() const;

    [[nodiscard]] std::string getNodeName() const override;

    std::vector<Literal *> evaluate(Ast &ast) override;

    bool operator==(const LiteralBool &rhs) const;

    bool operator!=(const LiteralBool &rhs) const;

    void addLiteralValue(std::string identifier, std::unordered_map<std::string, Literal *> &paramsMap) override;

    void setRandomValue(RandLiteralGen &rlg) override;

    void setValue(bool newValue);

    [[nodiscard]] std::string toString() const override;

    bool supportsDatatype(Datatype &datatype) override;

    void print(std::ostream &str) const override;

    bool supportsCircuitMode() override;
};

#endif //AST_OPTIMIZER_INCLUDE_LITERALBOOL_H
