#ifndef MASTER_THESIS_CODE_CLASS_H
#define MASTER_THESIS_CODE_CLASS_H


#include <vector>
#include "VarDecl.h"
#include "Function.h"

class Class : public AbstractStatement {
private:
    std::string name;
    std::string superclass;
    std::vector<Function> methods;
public:
    Class(const std::string &name, const std::string &superclass, const std::vector<Function> &methods);

    json toJson() const override;

    virtual void accept(Visitor &v) override;
};


#endif //MASTER_THESIS_CODE_CLASS_H
