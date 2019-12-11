#ifndef MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
#define MASTER_THESIS_CODE_FUNCTIONPARAMETER_H


#include "Variable.h"

class FunctionParameter : public Variable {
public:
    FunctionParameter(const std::string &identifier, std::string datatype);

    std::string datatype;

    json toJson() const;
};

/// Defines the JSON representation to be used for vector<FunctionParameter> objects.
void to_json(json &j, const FunctionParameter &param);


#endif //MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
