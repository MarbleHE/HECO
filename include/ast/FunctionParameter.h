
#ifndef MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
#define MASTER_THESIS_CODE_FUNCTIONPARAMETER_H


#include "Variable.h"

class FunctionParameter : public Variable {
    std::string datatype;
public:
    FunctionParameter(const std::string &identifier, std::string datatype);
};


#endif //MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
