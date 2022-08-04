
#include "heco/legacy_ast/ast_utilities/NodeUtils.h"
#include <unordered_map>

std::string NodeUtils::enumToString(const NodeType type)
{
    std::unordered_map<NodeType, std::string> typeToString = {
        { NodeType::NodeTypeAssignment, "Assignment" },
        { NodeType::NodeTypeBlock, "Block" },
        { NodeType::NodeTypeFunction, "Function" },
        { NodeType::NodeTypeFor, "For" },
        { NodeType::NodeTypeIf, "If" },
        { NodeType::NodeTypeReturn, "Return" },
        { NodeType::NodeTypeVariableDeclaration, "VariableDeclaration" },
        { NodeType::NodeTypeFunctionParameter, "FunctionParameter" },
        { NodeType::NodeTypeIndexAccess, "IndexAccess" },
        { NodeType::NodeTypeVariable, "Variable" },
        { NodeType::NodeTypeBinaryExpression, "BinaryExpression" },
        { NodeType::NodeTypeOperatorExpression, "OperatorExpression" },
        { NodeType::NodeTypeUnaryExpression, "UnaryExpression" },
        { NodeType::NodeTypeCall, "Call" },
        { NodeType::NodeTypeExpressionList, "ExpressionList" },
        { NodeType::NodeTypeLiteralBool, "LiteralBool" },
        { NodeType::NodeTypeLiteralChar, "LiteralChar" },
        { NodeType::NodeTypeLiteralInt, "LiteralInt" },
        { NodeType::NodeTypeLiteralFloat, "LiteralFloat" },
        { NodeType::NodeTypeLiteralDouble, "LiteralDouble" },
        { NodeType::NodeTypeLiteralString, "LiteralString" },
        { NodeType::NodeTypeTernaryOperator, "TernaryOperator" },
        { NodeType::NodeTypeLiteralBool, "LiteralBool" }
    };
    return typeToString.find(type)->second;
}

NodeType NodeUtils::stringToEnum(const std::string s)
{
    for (int nodeTypeInt = NodeTypeAssignment; nodeTypeInt != NodeTypeLastSymbol; nodeTypeInt++)
    {
        NodeType type = static_cast<NodeType>(nodeTypeInt);
        if (NodeUtils::enumToString(type) == s)
            return type;
    }
    return NodeTypeLastSymbol;
}

bool NodeUtils::isAbstractStatement(const std::string s)
{
    switch (NodeUtils::stringToEnum(s))
    {
    case NodeTypeAssignment:
    case NodeTypeBlock:
    case NodeTypeFunction:
    case NodeTypeFor:
    case NodeTypeIf:
    case NodeTypeReturn:
    case NodeTypeVariableDeclaration:
        return true;
    default:
        return false;
    }
}

bool NodeUtils::isAbstractTarget(const std::string s)
{
    switch (NodeUtils::stringToEnum(s))
    {
    case NodeTypeFunctionParameter:
    case NodeTypeIndexAccess:
    case NodeTypeVariable:
        return true;
    default:
        return false;
    }
}

bool NodeUtils::isAbstractExpression(const std::string s)
{
    switch (NodeUtils::stringToEnum(s))
    {
    case NodeTypeBinaryExpression:
    case NodeTypeOperatorExpression:
    case NodeTypeUnaryExpression:
    case NodeTypeCall:
    case NodeTypeExpressionList:
    case NodeTypeLiteralBool:
    case NodeTypeLiteralChar:
    case NodeTypeLiteralInt:
    case NodeTypeLiteralFloat:
    case NodeTypeLiteralDouble:
    case NodeTypeLiteralString:
    case NodeTypeTernaryOperator:
        return true;
    default:
        return NodeUtils::isAbstractTarget(s);
    }
}