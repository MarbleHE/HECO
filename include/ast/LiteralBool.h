#ifndef MASTER_THESIS_CODE_LITERALBOOL_H
#define MASTER_THESIS_CODE_LITERALBOOL_H


#include "Literal.h"

class LiteralBool : public Literal, public Node {
private:
    bool value;
public:
    explicit LiteralBool(bool value);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    bool isValue() const;

    std::string getTextValue() const;

    std::string getNodeName() const override;
};


#endif //MASTER_THESIS_CODE_LITERALBOOL_H
