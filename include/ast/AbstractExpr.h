#ifndef MASTER_THESIS_CODE_ABSTRACTEXPR_H
#define MASTER_THESIS_CODE_ABSTRACTEXPR_H

#include <string>
#include <nlohmann/json.hpp>
#include "../visitor/Visitor.h"


using json = nlohmann::json;


class AbstractExpr {
private:
    static LiteralInt *createParam(int i);

    static LiteralBool *createParam(bool b);

    static LiteralString *createParam(const std::string &str);

    static AbstractExpr *createParam(AbstractExpr *abstractExpr);

public:
    virtual ~AbstractExpr() = default;

    virtual std::string toString() const;

    virtual json toJson() const;

    virtual void accept(Visitor &v);

};

std::ostream &operator<<(std::ostream &outs, AbstractExpr &obj);


#endif //MASTER_THESIS_CODE_ABSTRACTEXPR_H
