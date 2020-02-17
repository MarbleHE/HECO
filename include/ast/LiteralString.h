#ifndef AST_OPTIMIZER_INCLUDE_LITERALSTRING_H
#define AST_OPTIMIZER_INCLUDE_LITERALSTRING_H

#include <string>
#include "Literal.h"
#include <map>

class LiteralString : public Literal {
private:
    std::string value;

    Node *createClonedNode(bool keepOriginalUniqueNodeId) override;

protected:
    void print(std::ostream &str) const override;

public:
    explicit LiteralString(std::string value);

    ~LiteralString() override;

    [[nodiscard]] json toJson() const override;

    [[nodiscard]] const std::string &getValue() const;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    std::vector<Literal *> evaluate(Ast &ast) override;

    bool operator==(const LiteralString &rhs) const;

    bool operator!=(const LiteralString &rhs) const;

    void addLiteralValue(std::string identifier, std::unordered_map<std::string, Literal *> &paramsMap) override;

    void setValue(const std::string &newValue);

    void setRandomValue(RandLiteralGen &rlg) override;

    [[nodiscard]] std::string toString() const override;

    bool supportsCircuitMode() override;

    bool supportsDatatype(Datatype &datatype) override;
};

#endif //AST_OPTIMIZER_INCLUDE_LITERALSTRING_H
