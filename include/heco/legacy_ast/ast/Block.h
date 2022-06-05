#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_BLOCK_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_BLOCK_H_

#include <vector>
#include <string>
#include "AbstractStatement.h"

/// This class is a simple wrapper around a list of statements
/// Most importantly, it can be used to indicate scopes
class Block : public AbstractStatement {
 private:
  /// Stores the Statements in the Block
  std::vector<std::unique_ptr<AbstractStatement>> statements;

  /// Creates a deep copy of the current node
  /// Should be used only by Nodes' clone()
  /// \return a copy of the current node
  Block *clone_impl(AbstractNode* parent) const override;

 public:
  /// Destructor
  ~Block() override;

  /// Create an empty Block
  Block();

  /// Create a Block with a single statement
  /// \param statement Block will contain only this statement (Block takes ownership)
  explicit Block(std::unique_ptr<AbstractStatement> statement);

  /// Create a Block from a vector of statements
  /// \param vectorOfStatements Block takes ownership of both statements and vector
  explicit Block(std::vector<std::unique_ptr<AbstractStatement>> &&vectorOfStatements);

  /// Copy constructor
  /// \param other Block to copy
  Block(const Block &other);

  /// Move constructor
  /// \param other Block to copy
  Block(Block &&other) noexcept;

  /// Copy assignment
  /// \param other Block to copy
  /// \return This object
  Block &operator=(const Block &other);

  /// Move assignment
  /// \param other Block to move
  /// \return This object
  Block &operator=(Block &&other) noexcept;

  /// Deep copy of the current node
  /// \return A deep copy of the current node
  std::unique_ptr<Block> clone(AbstractNode* parent = nullptr) const;

  /// Is the current Block (logically) empty?
  /// \return true iff the block contains no (non-null) statements
  bool isEmpty();

  /// Does the current Block contain null statements?
  bool hasNullStatements();

  /// Get (references to) the vector of statements
  /// Used for lower-level interactions (e.g. removing statements by nulling them)
  /// \return (references to) the vector of statements
  std::vector<std::unique_ptr<AbstractStatement>>& getStatementPointers();

  /// Get (a vector of references to) all (non-null) statements
  /// Since std::vector cannot directly handle references,
  /// a wrapper is used, but this can be used exactly like
  /// std::vector<AbstractStatement&> could, if it were possible
  /// \return Vector of (references to) all non-null statements
  std::vector<std::reference_wrapper<AbstractStatement>> getStatements();

  /// Get (a vector of const references to) all (non-null) statements
  /// Since std::vector cannot directly handle references,
  /// a wrapper is used, but this can be used exactly like
  /// std::vector<const AbstractStatement&> could, if it were possible
  /// \return Vector of (const references to) all non-null statements
  std::vector<std::reference_wrapper<const AbstractStatement>> getStatements() const;

  /// Add a statement as the last statement in the Block
  /// \param statement Statement to be appended, Block will take ownership
  void appendStatement(std::unique_ptr<AbstractStatement> statement);

  /// Prepend a statement as the new first statement in the Block
  /// \param statement Statement to be prepended, Block will take ownership
  void prependStatement(std::unique_ptr<AbstractStatement> statement);

  /// Removes any potential nullptrs from the statements vector
  void removeNullStatements();

  /// Create a Block node from a nlohmann::json representation of this node.
  /// \return unique_ptr to a new Block node
  static std::unique_ptr<Block> fromJson(nlohmann::json j);

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
template<typename T>
class BlockIteratorImpl : public BaseIteratorImpl<T> {
 private:
  // Select const Block / Block depending on whether or not T is const/non-const
  typedef typename std::conditional<std::is_const<T>::value, const Block, Block>::type N;

  // Select const iterator / iterator depending on whether or not T is const/non-const
  typedef typename std::conditional<std::is_const<T>::value,
                                    std::vector<std::unique_ptr<AbstractStatement>>::const_iterator,
                                    std::vector<std::unique_ptr<AbstractStatement>>::iterator>::type I;

  /// The node object that this iterator belongs to
  N &node;

  /// Vector iterator
  I it;

  /// Vector iterator indicating end - this is required to avoid dereferencing past the end in increment()
  I end;

 public:

  BlockIteratorImpl(N &node, I it, I end) : node(node), it(it), end(end) {};

  const T &getNode() const override {
    return node;
  }

  T &getNode() override {
    return node;
  }

  void increment() override {
    // First, increment the underlying iterator normally
    ++it;
    // Then, keep advancing until we stop seeing nullptrs or hit the end
    while (it!=end && *it==nullptr) {
      ++it;
    }
  }

  bool equal(const BaseIteratorImpl<T> &other) override {
    if (node==other.getNode()) {
      auto otherNodePtr = dynamic_cast<const BlockIteratorImpl *>(&other);
      assert(otherNodePtr); // If the other node has the same type, the Iterator must be the same type, too
      return (it==otherNodePtr->it);
    } else {
      return false;
    }
  }

  T &operator*() override {
    //simply forward to vector iterator and follow the unique_ptr
    return **it;
  }

  std::unique_ptr<BaseIteratorImpl<T>> clone() override {
    return std::make_unique<BlockIteratorImpl>(node, it, end);
  }
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_BLOCK_H_
