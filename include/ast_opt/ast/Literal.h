#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERAL_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERAL_H_

#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/visitor/IVisitor.h"

template<typename T>
class Literal;

typedef Literal<bool> LiteralBool;
typedef Literal<char> LiteralChar;
typedef Literal<int> LiteralInt;
typedef Literal<float> LiteralFloat;
typedef Literal<double> LiteralDouble;
typedef Literal<std::string> LiteralString;

/// Literals contain a scalar value (of type bool, char, int, float, double, string, etc)
template<typename T>
class Literal : public AbstractExpression {
 private:
  T value;

  /// Creates a deep copy of the current node
  /// Should be used only by Literal::clone()
  /// \return a copy of the current node
  Literal *clone_impl() const override {
    return new Literal(*this);
  }

 public:
  /// Destructor
  ~Literal() override = default;

  //TODO: Template magic to decide if there's a default value (free function?)
  /// Default constructor
  /// Creates a scalar boolean of value FALSE
  //Literal();

  /// Creates a Literal with value value
  /// \param value value to store in this Literal
  explicit Literal(T value) : value(value) {};

  /// Copy constructor
  /// \param other Literal to copy
  Literal(const Literal &other) : value(other.value) {};

  /// Move constructor
  /// \param other Literal to copy
  Literal(Literal &&other) : value(std::move(other.value)) {};

  /// Copy assignment
  /// \param other Literal to copy
  /// \return This object
  Literal &operator=(const Literal &other) {
    value = other.value;
    return *this;
  }

  /// Move assignment
  /// \param other Literal to move
  /// \return This object
  Literal &operator=(Literal &&other)  noexcept {
    value = std::move(other.value);
    return *this;
  }

  /// Get value
  /// \return The value stored in this Literal
  T getValue() const {
    return value;
  }

  /// Sets the value of this Literal to newValue
  /// \param newValue
  void setValue(T newValue) {
    value = newValue;
  };

  /// Sets the value of this Literal to newValue
  /// \param newValue will be moved to this Literal
  void setMatrix(T &&newValue) {
    value = std::move(newValue);
  };

#include "ast_opt/utilities/warning_hidingNonVirtualFunction_prologue.h"

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<Literal> clone() const { /* intentionally hiding */
    return std::unique_ptr<Literal>(clone_impl());
  }

#include "ast_opt/utilities/warning_hidingNonVirtualFunction_epilogue.h"



  ///////////////////////////////////////////////
  ////////// AbstractNode Interface /////////////
  ///////////////////////////////////////////////

  void accept(IVisitor &v) override {
    v.visit(*this);
  }

  iterator begin() override {
    return iterator(std::make_unique<EmptyIteratorImpl<AbstractNode>>(*this));
  }

  const_iterator begin() const override {
    return const_iterator(std::make_unique<EmptyIteratorImpl<const AbstractNode>>(*this));
  }

  iterator end() override {
    return iterator(std::make_unique<EmptyIteratorImpl<AbstractNode>>(*this));
  }

  const_iterator end() const override {
    return const_iterator(std::make_unique<EmptyIteratorImpl<const AbstractNode>>(*this));
  }

  size_t countChildren() const override {
    return 0;
  }

  nlohmann::json toJson() const override {
    nlohmann::json j;
    j["type"] = getNodeType();
    j["value"] = value;
    return j;
  }

  std::string toString(bool printChildren) const override {
    std::stringstream ss;
    ss << std::boolalpha; // Ensure that bools are printed as true/false instead of 1/0
    ss << value; // Since there is no toString on builtins
    return toStringHelper(printChildren, {ss.str()});
  }

 protected:
  /// Returns the node's type, which is the name of the object in the AST.
  /// If "pretty" names are desired, a specialization of this function for the type must be provided.
  /// \return "Literal<T>" (possibly name-mangled)
  std::string getNodeType() const override {
    std::stringstream ss;
    ss << "Literal<" << typeid(T).name() << ">";
    return ss.str();
  }

};


// Pretty "getNodeType()" specializations for the common types:

template<>
inline std::string Literal<bool>::getNodeType() const { return "LiteralBool"; };

template<>
inline std::string Literal<char>::getNodeType() const { return "LiteralChar"; };

template<>
inline std::string Literal<int>::getNodeType() const { return "LiteralInt"; };

template<>
inline std::string Literal<float>::getNodeType() const { return "LiteralFloat"; };

template<>
inline std::string Literal<double>::getNodeType() const { return "LiteralBool"; };

template<>
inline std::string Literal<std::string>::getNodeType() const { return "LiteralString"; };

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERAL_H_
