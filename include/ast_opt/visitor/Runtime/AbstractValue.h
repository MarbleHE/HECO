#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_

class AbstractValue {
 protected:
  // make sure that class is abstract, i.e., cannot be instantiated
  AbstractValue() = default;

 public:
  /// Default destructor.
  virtual ~AbstractValue() = default;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTVALUE_H_
