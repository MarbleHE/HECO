#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTION_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTION_H_

#include <vector>
#include <string>
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/utilities/Datatype.h"

/// A function has a return type, name, a list of parameters and a body
class Function : public AbstractStatement {
 private:
  /// Return type of the function
  Datatype return_type;

  /// Name of the function
  std::string identifier;

  /// Function Parameters
  std::vector<std::unique_ptr<FunctionParameter>> parameters;

  /// Body of the function
  std::unique_ptr<Block> body;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  Function *clone_impl() const override;

 public:
  /// Destructor
  ~Function() override;

  /// Create a Function
  /// \param return_type Return type of the function
  /// \param identifier  Name of the function
  /// \param parameters List of function parameters
  /// \param body Body of the function
  Function(Datatype return_type,
           std::string identifier,
           std::vector<std::unique_ptr<FunctionParameter>> parameters,
           std::unique_ptr<Block> body);

  /// Copy constructor
  /// \param other Function to copy
  Function(const Function &other);

  /// Move constructor
  /// \param other Function to copy
  Function(Function &&other) noexcept;

  /// Copy assignment
  /// \param other Function to copy
  /// \return This object
  Function &operator=(const Function &other);

  /// Move assignment
  /// \param other Function to move
  /// \return This object
  Function &operator=(Function &&other) noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<Function> clone() const;

  /// Returns (a copy of) the return type of the Function
  /// \return the return type of the Function
  Datatype getReturnType() const;

  /// Returns (a copy of) the name of the Function
  /// \return the name of the Function
  std::string getIdentifier() const;

  /// Get (a vector of references to) all (non-null) parameters
  /// Since std::vector cannot directly handle references,
  /// a wrapper is used, but this can be used exactly like
  /// std::vector<AbstractStatement&> could, if it were possible
  /// \return Vector of (references to) all non-null parameters
  std::vector<std::reference_wrapper<const FunctionParameter>> getParameters() const;

  /// Get (a vector of references to) all (non-null) parameters
  /// Since std::vector cannot directly handle references,
  /// a wrapper is used, but this can be used exactly like
  /// std::vector<AbstractStatement&> could, if it were possible
  /// \return Vector of (references to) all non-null parameters
  std::vector<std::reference_wrapper<FunctionParameter>> getParameters();

  /// Checks whether the Function has a body
  /// \return true iff the Function has a body
  bool hasBody() const;

  /// Returns the body, if it exists
  /// \return the Function's body
  /// \throws std::runtime_error if no body exists
  const Block &getBody() const;

  /// Returns the body, if it exists
  /// \return the Function's body
  /// \throws std::runtime_error if no body exists
  Block &getBody();

  ///////////////////////////////////////////////
  ////////// AbstractNode Interface /////////////
  ///////////////////////////////////////////////
  void accept(IVisitor &v) override;
  iterator begin() override;
  const_iterator begin() const override;
  iterator end() override;
  const_iterator end() const override;
  size_t countChildren() const override;
  nlohmann::json toJson() const override;
  std::string toString(bool printChildren) const override;
 protected:
  std::string getNodeType() const override;

};

template<typename T>
class FunctionIteratorImpl : public BaseIteratorImpl<T> {
 private:
  // Select const Function / Function depending on whether or not T is const/non-const
  typedef typename std::conditional<std::is_const<T>::value, const Function, Function>::type N;

  // Select const iterator / iterator depending on whether or not T is const/non-const
  typedef typename std::conditional<std::is_const<T>::value,
                                    std::vector<std::unique_ptr<FunctionParameter>>::const_iterator,
                                    std::vector<std::unique_ptr<FunctionParameter>>::iterator>::type I;

  /// The node object that this iterator belongs to
  N &node;

  /// Parameter vector iterator
  I it;

  /// Parameter vector iterator indicating end - this is required to avoid dereferencing past the end in increment()
  I end;

  /// Does iterator point past the parameter vector? I.e. if we reached end, then this is 1,
  /// the "end" FunctionIteratorImpl has it == end & positions_beyond_parameter_vector = 2 (or 1 if there is no body)
  size_t positions_beyond_parameter_vector;

 public:

  FunctionIteratorImpl(N &node, I it, I end, size_t positions_beyond_parameter_vector)
      : node(node), it(it), end(end), positions_beyond_parameter_vector(positions_beyond_parameter_vector) {};

  const T &getNode() const override {
    return node;
  }

  T &getNode() override {
    return node;
  }

  void increment() override {
    if (it==end) {
      ++positions_beyond_parameter_vector;
    } else {
      // First, increment the underlying iterator normally
      ++it;
      // Then, keep advancing until we stop seeing nullptrs or hit the end
      while (it!=end && *it==nullptr) {
        ++it;
      }
      // If we've hit end, we need to set positions_beyond_parameter_vector
      if (it==end) {
        positions_beyond_parameter_vector = 1;
      }
    }
  }

  bool equal(const BaseIteratorImpl<T> &other) override {
    if (node==other.getNode()) {
      auto otherNodePtr = dynamic_cast<const FunctionIteratorImpl *>(&other);
      assert(otherNodePtr); // If the other node has the same type, the Iterator must be the same type, too
      return (it==otherNodePtr->it
          && positions_beyond_parameter_vector==otherNodePtr->positions_beyond_parameter_vector);
    } else {
      return false;
    }
  }

  T &operator*() override {
    //simply forward to vector iterator and follow the unique_ptr
    if (it!=end) {
      return **it;
    } else if (positions_beyond_parameter_vector==1) {
      return node.getBody(); // if there is no body, this call is already UB so throwing an exception is fine
    } else {
      assert(0); // Undefined Behaviour, might as well crash
      return **it; // to avoid "not all paths return a value"
    }
  }

  std::unique_ptr<BaseIteratorImpl<T>> clone() override {
    return std::make_unique<FunctionIteratorImpl>(node, it, end, positions_beyond_parameter_vector);
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTION_H_
