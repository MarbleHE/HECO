#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_CALL_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_CALL_H_

#include "heco/ast/AbstractExpression.h"

// We don't allow side effects, so there's no point in having a stmt with only one call
class Call : public AbstractExpression
{
private:
  /// Name of the Call being called
  std::string identifier;

  /// List of Call arguments
  std::vector<std::unique_ptr<AbstractExpression>> arguments;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  Call *clone_impl(AbstractNode *parent) const override;

public:
  /// Destructor
  ~Call() override;

  /// Create a Call
  /// \param identifier  Name of the function being called
  /// \param arguments List of arguments
  Call(std::string identifier, std::vector<std::unique_ptr<AbstractExpression>> &&arguments);

  /// Copy constructor
  /// \param other Call to copy
  Call(const Call &other);

  /// Move constructor
  /// \param other Call to copy
  Call(Call &&other) noexcept;

  /// Copy assignment
  /// \param other Call to copy
  /// \return This object
  Call &operator=(const Call &other);

  /// Move assignment
  /// \param other Call to move
  /// \return This object
  Call &operator=(Call &&other) noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<Call> clone(AbstractNode *parent = nullptr) const;

  /// Returns (a copy of) the name of the Call being called
  /// \return the name of the Call being called
  std::string getIdentifier() const;

  /// Get (a vector of references to) all (non-null) arguments
  /// Since std::vector cannot directly handle references,
  /// a wrapper is used, but this can be used exactly like
  /// std::vector<AbstractExpression&> could, if it were possible
  /// \return Vector of (references to) all non-null arguments
  std::vector<std::reference_wrapper<const AbstractExpression>> getArguments() const;

  /// Get (a vector of references to) all (non-null) parameters
  /// Since std::vector cannot directly handle references,
  /// a wrapper is used, but this can be used exactly like
  /// std::vector<AbstractExpression&> could, if it were possible
  /// \return Vector of (references to) all non-null arguments
  std::vector<std::reference_wrapper<AbstractExpression>> getArguments();

  /// Create a Call node from a nlohmann::json representation of this node.
  /// \return unique_ptr to a new Call node
  static std::unique_ptr<Call> fromJson(nlohmann::json j);

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

// Designed to be instantiated only with T = (const) AbstractNode
// This essentially just wraps the vector iterator
// jumping over null positions on operator++
template <typename T>
class CallIteratorImpl : public BaseIteratorImpl<T>
{
private:
  // Select const Call / Call depending on whether or not T is const/non-const
  typedef typename std::conditional<std::is_const<T>::value, const Call, Call>::type N;

  // Select const iterator / iterator depending on whether or not T is const/non-const
  typedef typename std::conditional<std::is_const<T>::value,
                                    std::vector<std::unique_ptr<AbstractExpression>>::const_iterator,
                                    std::vector<std::unique_ptr<AbstractExpression>>::iterator>::type I;

  /// The node object that this iterator belongs to
  N &node;

  /// Vector iterator
  I it;

  /// Vector iterator indicating end - this is required to avoid dereferencing past the end in increment()
  I end;

public:
  CallIteratorImpl(N &node, I it, I end) : node(node), it(it), end(end){};

  const T &getNode() const override
  {
    return node;
  }

  T &getNode() override
  {
    return node;
  }

  void increment() override
  {
    // First, increment the underlying iterator normally
    ++it;
    // Then, keep advancing until we stop seeing nullptrs or hit the end
    while (it != end && *it == nullptr)
    {
      ++it;
    }
  }

  bool equal(const BaseIteratorImpl<T> &other) override
  {
    if (node == other.getNode())
    {
      auto otherNodePtr = dynamic_cast<const CallIteratorImpl *>(&other);
      assert(otherNodePtr); // If the other node has the same type, the Iterator must be the same type, too
      return (it == otherNodePtr->it);
    }
    else
    {
      return false;
    }
  }

  T &operator*() override
  {
    // simply forward to vector iterator and follow the unique_ptr
    return **it;
  }

  std::unique_ptr<BaseIteratorImpl<T>> clone() override
  {
    return std::make_unique<CallIteratorImpl>(node, it, end);
  }
};
#endif // AST_OPTIMIZER_INCLUDE_AST_OPT_AST_CALL_H_
