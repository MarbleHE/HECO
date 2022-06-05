#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_NODEUTILS_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_NODEUTILS_H_

#include <string>
#include <memory>
#include <stdexcept>
#include "heco/ast_parser/Errors.h"

enum NodeType : unsigned char
{
  // AbstractStatement
  NodeTypeAssignment = 0,
  NodeTypeBlock,
  NodeTypeFunction,
  NodeTypeFor,
  NodeTypeIf,
  NodeTypeReturn,
  NodeTypeVariableDeclaration,

  // AbstractExpression -> AbstractTarget
  NodeTypeFunctionParameter,
  NodeTypeIndexAccess,
  NodeTypeVariable,

  // AbstractExpression
  NodeTypeBinaryExpression,
  NodeTypeOperatorExpression,
  NodeTypeUnaryExpression,
  NodeTypeCall,
  NodeTypeExpressionList,
  NodeTypeLiteralBool,
  NodeTypeLiteralChar,
  NodeTypeLiteralInt,
  NodeTypeLiteralFloat,
  NodeTypeLiteralDouble,
  NodeTypeLiteralString,
  NodeTypeTernaryOperator,

  // Symbol to mark the end of enum types and not actually a node type
  NodeTypeLastSymbol
};

/// Cast a unique_ptr of (node) type S to (node) subtype T
// TODO: change RuntimeVisitor to also use this function, once it's merged.
template <typename S, typename T>
std::unique_ptr<T> castUniquePtr(std::unique_ptr<S> &&source)
{
  if (dynamic_cast<T *>(source.get()))
  {
    return std::unique_ptr<T>(dynamic_cast<T *>(source.release()));
  }
  else
  {
    throw stork::runtime_error("castUniquePtr failed: Cannot cast given unique_ptr from type " + std::string(typeid(S).name()) + " to type " + std::string(typeid(T).name()) + ".");
  }
}

class NodeUtils
{
public:
  /// Convert a NodeType enum to string
  static std::string enumToString(const NodeType type);

  /// Convert string to enum, returns the enum or LastSymbol if no enum with this name was found.
  static NodeType stringToEnum(const std::string s);

  /// Return true when the given string represents a subclass of an AbstractStatement node
  static bool isAbstractStatement(const std::string s);

  /// Return true when the given string represents a subclass of an AbstractTarget node
  static bool isAbstractTarget(const std::string s);

  /// Return true when the given string represents a subclass of an AbstractExpression node
  static bool isAbstractExpression(const std::string s);
};

#endif // AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_NODEUTILS_H_
