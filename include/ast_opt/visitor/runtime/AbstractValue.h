#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_

class AbstractValue {
 protected:
  // make sure that class is abstract, i.e., cannot be instantiated
  AbstractValue() = default;

 public:
  /// Default destructor.
  virtual ~AbstractValue() = default;

  virtual void add(AbstractValue &other) = 0;

  virtual void subtract(AbstractValue &other) = 0;

  virtual void multiply(AbstractValue &other) = 0;

  virtual void divide(AbstractValue &other) = 0;

  virtual void modulo(AbstractValue &other) = 0;

  virtual void logicalAnd(AbstractValue &other) = 0;

  virtual void logicalOr(AbstractValue &other) = 0;

  virtual void logicalLess(AbstractValue &other) = 0;

  virtual void logicalLessEqual(AbstractValue &other) = 0;

  virtual void logicalGreater(AbstractValue &other) = 0;

  virtual void logicalGreaterEqual(AbstractValue &other) = 0;

  virtual void logicalEqual(AbstractValue &other) = 0;

  virtual void logicalNotEqual(AbstractValue &other) = 0;

  virtual void logicalNot() = 0;

  virtual void bitwiseAnd(AbstractValue &other) = 0;

  virtual void bitwiseXor(AbstractValue &other) = 0;

  virtual void bitwiseOr(AbstractValue &other) = 0;

  virtual void bitwiseNot() = 0;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_
