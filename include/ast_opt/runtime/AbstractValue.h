#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_

class AbstractValue {
 protected:
  // make sure that class is abstract, i.e., cannot be instantiated
  AbstractValue() = default;

 public:
  /// Default destructor.
  virtual ~AbstractValue() = default;

  virtual void add_inplace(const AbstractValue &other) = 0;

  virtual void subtract_inplace(const AbstractValue &other) = 0;

  virtual void multiply_inplace(const AbstractValue &other) = 0;

  virtual void divide_inplace(const AbstractValue &other) = 0;

  virtual void modulo_inplace(const AbstractValue &other) = 0;

  virtual void logicalAnd_inplace(const AbstractValue &other) = 0;

  virtual void logicalOr_inplace(const AbstractValue &other) = 0;

  virtual void logicalLess_inplace(const AbstractValue &other) = 0;

  virtual void logicalLessEqual_inplace(const AbstractValue &other) = 0;

  virtual void logicalGreater_inplace(const AbstractValue &other) = 0;

  virtual void logicalGreaterEqual_inplace(const AbstractValue &other) = 0;

  virtual void logicalEqual_inplace(const AbstractValue &other) = 0;

  virtual void logicalNotEqual_inplace(const AbstractValue &other) = 0;

  virtual void logicalNot_inplace() = 0;

  virtual void bitwiseAnd_inplace(const AbstractValue &other) = 0;

  virtual void bitwiseXor_inplace(const AbstractValue &other) = 0;

  virtual void bitwiseOr_inplace(const AbstractValue &other) = 0;

  virtual void bitwiseNot_inplace() = 0;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_
