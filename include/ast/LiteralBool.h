#ifndef MASTER_THESIS_CODE_LITERALBOOL_H
#define MASTER_THESIS_CODE_LITERALBOOL_H


#include "Literal.h"

class LiteralBool : public Literal, public Node {
private:
    bool value;
public:
    explicit LiteralBool(bool value);

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] bool isValue() const;

    [[nodiscard]] std::string getTextValue() const;

    [[nodiscard]] std::string getNodeName() const override;
};


#endif //MASTER_THESIS_CODE_LITERALBOOL_H
