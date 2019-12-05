#ifndef MASTER_THESIS_CODE_CLASS_H
#define MASTER_THESIS_CODE_CLASS_H


#include <vector>
#include "VarDecl.h"
#include "Function.h"

class Class : public AbstractStatement {
    std::string name;

private:
    Class* superclass;
    std::vector<Function> methods;
public:
    Class(std::string name, Class *superclass, std::vector<Function> methods);
};


#endif //MASTER_THESIS_CODE_CLASS_H
