#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_NODEUTILS_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_NODEUTILS_H_

// TODO: write function to detect target, expression, or statement
enum NodeType {
  // AbstractStatement
  NodeTypeAssignment, NodeTypeBlock, NodeTypeFor, NodeTypeIf, NodeTypeReturn, NodeTypeVariableDeclaration,

  // AbstractExpression -> AbstractTarget
  NodeTypeFunctionParameter, NodeTypeIndexAccess, NodeTypeVariable,

  // AbstractExpression
  NodeTypeBinaryExpression, NodeTypeOperatorExpression, NodeTypeUnaryExpression, NodeTypeCall,
  NodeTypeExpressionList, NodeTypeLiteralBool, NodeTypeLiteralChar, NodeTypeLiteralInt, NodeTypeLiteralFloat,
  NodeTypeLiteralDouble, NodeTypeLiteralString, NodeTypeTernaryOperator,

  // Symbol to mark the end of enum types and not actually a node type
  NodeTypeLastSymbol
};

class NodeUtils {
 public:

  /// Convert a NodeType enum to string
  static std::string enumToString(const NodeType type) {
    std::unordered_map<NodeType, std::string> typeToString = {
        {NodeType::NodeTypeAssignment, "Assignment"},
        {NodeType::NodeTypeBlock, "Block"},
        {NodeType::NodeTypeFor, "For"},
        {NodeType::NodeTypeIf, "If"},
        {NodeType::NodeTypeReturn, "Return"},
        {NodeType::NodeTypeVariableDeclaration, "VariableDeclaration"},
        {NodeType::NodeTypeFunctionParameter, "FunctionParameter"},
        {NodeType::NodeTypeIndexAccess, "IndexAccess"},
        {NodeType::NodeTypeVariable, "Variable"},
        {NodeType::NodeTypeBinaryExpression, "BinaryExpression"},
        {NodeType::NodeTypeOperatorExpression, "OperatorExpression"},
        {NodeType::NodeTypeUnaryExpression, "UnaryExpression"},
        {NodeType::NodeTypeCall, "Call"},
        {NodeType::NodeTypeExpressionList, "ExpressionList"},
        {NodeType::NodeTypeLiteralBool, "LiteralBool"},
        {NodeType::NodeTypeLiteralChar, "LiteralChar"},
        {NodeType::NodeTypeLiteralInt, "LiteralInt"},
        {NodeType::NodeTypeLiteralFloat, "LiteralFloat"},
        {NodeType::NodeTypeLiteralDouble, "LiteralDouble"},
        {NodeType::NodeTypeLiteralString, "LiteralString"},
        {NodeType::NodeTypeTernaryOperator, "TernaryOperator"},
        {NodeType::NodeTypeLiteralBool, "LiteralBool"}
    };
    return typeToString.find(type)->second;
  }

  /// Convert string to enum, returns the enum or LastSymbol if no enum with this name was found.
  static NodeType stringToEnum(const std::string s) {
    for ( int nodeTypeInt = NodeTypeAssignment; nodeTypeInt != NodeTypeLastSymbol; nodeTypeInt++ ) {
      NodeType type = static_cast<NodeType>(nodeTypeInt);
      if (NodeUtils::enumToString(type)  == s)
        return type;
    }
    return NodeTypeLastSymbol;
  }

  /// Return true when the given string represents a subclass of an AbstractStatement node
  static bool isAbstractStatement(const std::string s) {
    switch (NodeUtils::stringToEnum(s)) {
      case NodeTypeAssignment:
      case NodeTypeBlock:
      case NodeTypeFor:
      case NodeTypeIf:
      case NodeTypeReturn:
      case NodeTypeVariableDeclaration:
        return true;
      default:
        return false;
    }
  }

  /// Return true when the given string represents a subclass of an AbstractTarget node
  static bool isAbstractTarget(const std::string s) {
    switch (NodeUtils::stringToEnum(s)) {
      case NodeTypeFunctionParameter:
      case NodeTypeIndexAccess:
      case NodeTypeVariable:
        return true;
      default:
        return false;
    }
  }

  /// Return true when the given string represents a subclass of an AbstractExpression node
  static bool isAbstractExpression(const std::string s) {
    switch (NodeUtils::stringToEnum(s)) {
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
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_NODEUTILS_H_
