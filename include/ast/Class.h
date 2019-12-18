#ifndef MASTER_THESIS_CODE_CLASS_H
#define MASTER_THESIS_CODE_CLASS_H


#include <vector>
#include "VarDecl.h"
#include "Function.h"

class Class : public AbstractStatement, public Node {
private:
    std::string name;
    std::string superclass;
    std::vector<Function> methods;
public:
    Class(std::string name, std::string superclass, std::vector<Function> methods);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    const std::string &getName() const;

    const std::string &getSuperclass() const;

    const std::vector<Function> &getMethods() const;

    std::string getNodeName() const override;
};


#endif //MASTER_THESIS_CODE_CLASS_H
