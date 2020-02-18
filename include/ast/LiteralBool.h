#ifndef AST_OPTIMIZER_INCLUDE_LITERALBOOL_H
#define AST_OPTIMIZER_INCLUDE_LITERALBOOL_H

#include "AbstractLiteral.h"
#include <string>
#include <map>

class LiteralBool : public AbstractLiteral {
private:
    bool value;
public:
    explicit LiteralBool(bool value);

    ~LiteralBool() override;

    LiteralBool *clone(bool keepOriginalUniqueNodeId) override;

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] bool getValue() const;

    [[nodiscard]] std::string getTextValue() const;

    [[nodiscard]] std::string getNodeName() const override;

    bool operator==(const LiteralBool &rhs) const;

    bool operator!=(const LiteralBool &rhs) const;

    void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

    void setRandomValue(RandLiteralGen &rlg) override;

    void setValue(bool newValue);

    [[nodiscard]] std::string toString() const override;

    bool supportsDatatype(Datatype &datatype) override;

    void print(std::ostream &str) const override;

    bool supportsCircuitMode() override;
};

#endif //AST_OPTIMIZER_INCLUDE_LITERALBOOL_H
