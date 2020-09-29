#include <queue>
#include <iostream>
#include "ast_opt/ast/AbstractTarget.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/Vectorizer.h"

////////////////////////////////////////////
////        BatchingConstraint          ////
////////////////////////////////////////////
BatchingConstraint::BatchingConstraint(int slot, const ScopedIdentifier &identifier)
    : slot(slot), identifier(identifier) {}

int BatchingConstraint::getSlot() const {
  return slot;
}
void BatchingConstraint::setSlot(int slot_) {
  slot = slot_;
}
const ScopedIdentifier &BatchingConstraint::getIdentifier() const {
  return identifier;
}
void BatchingConstraint::setIdentifier(const ScopedIdentifier &identifier_) {
  identifier = identifier_;
}
bool BatchingConstraint::hasTargetSlot() const {
  return getSlot()!=-1;
}

////////////////////////////////////////////
////           ComplexValue             ////
////////////////////////////////////////////
ComplexValue::ComplexValue(AbstractExpression &) {
  //TODO: Implement ComplexValue Ctor
}

BatchingConstraint &ComplexValue::getBatchingConstraint() {
  //TODO: Implement ComplexValue::getBatchingConstraint
  return batchingConstraint;
}
void ComplexValue::merge(ComplexValue value) {
  //TODO: Implement
}
std::vector<std::unique_ptr<AbstractStatement>> ComplexValue::statementsToExecutePlan() {
  //TODO: Implement
  return {};
}

////////////////////////////////////////////
////         VariableValueMap           ////
////////////////////////////////////////////
void VariableValueMap::add(ScopedIdentifier s, ComplexValue &cv) {
  map.insert({s, cv});
  changed.insert(s);
}

const ComplexValue &VariableValueMap::get(const ScopedIdentifier &s) const {
  return map.find(s)->second;
}

ComplexValue &VariableValueMap::getToModify(const ScopedIdentifier &s) {
  changed.insert(s);
  return map.find(s)->second;
}

ComplexValue &VariableValueMap::take(const ScopedIdentifier &s) {
  auto it = map.find(s);
  ComplexValue &cv = it->second;
  map.erase(it);
  auto changed_it = changed.find(s);
  if (changed_it!=changed.end()) {
    changed.erase(changed_it);
  }
  return cv;
}
void VariableValueMap::update(const ScopedIdentifier &s, ComplexValue &cv) {
  map.find(s)->second = cv;
  changed.insert(s);
}
bool VariableValueMap::has(const ScopedIdentifier &s) {
  return map.find(s)!=map.end();
}
void VariableValueMap::resetChangeFlags() {
  changed.clear();
}
std::unordered_set<ScopedIdentifier> VariableValueMap::changedEntries() const {
  return changed;
}


////////////////////////////////////////////
////          SpecialVectorizer         ////
////////////////////////////////////////////

void SpecialVectorizer::visit(Block &elem) {
  ScopedVisitor::enterScope(elem);
  variableValues.resetChangeFlags();

  for (auto &p: elem.getStatementPointers()) {
    p->accept(*this);
    if (delete_flag) { p.reset(); }
    delete_flag = false;
  }
  elem.removeNullStatements();

  // TODO: Emit all relevant assignments again!
  for (auto &scopedID: variableValues.changedEntries()) {
    auto &cv = variableValues.take(scopedID);
    for (auto &statement : cv.statementsToExecutePlan()) {
      elem.appendStatement(std::move(statement));
    }
  }
  variableValues.resetChangeFlags();
  ScopedVisitor::exitScope();
}

void SpecialVectorizer::visit(Assignment &elem) {

  /// current scope
  auto &scope = getCurrentScope();

  /// target of the assignment
  AbstractTarget &target = elem.getTarget();
  ScopedIdentifier targetID;
  BatchingConstraint targetBatchingConstraint;

  // We currently assume that the target has either the form <Variable> or <Variable>[<LiteralInt>]
  if (target.countChildren()==0) {
    auto variable = dynamic_cast<Variable &>(target);
    auto id = variable.getIdentifier();
    targetID = scope.resolveIdentifier(id);
    if (constraints.find(targetID)!=constraints.end()) {
      auto t = constraints.find(targetID)->second.getSlot();
      targetBatchingConstraint = BatchingConstraint(t, targetID);
    }
  } else {
    auto indexAccess = dynamic_cast<IndexAccess &>(target);
    auto variable = dynamic_cast<Variable &>(indexAccess.getTarget());
    auto index = dynamic_cast<LiteralInt &>(indexAccess.getIndex());
    targetID = scope.resolveIdentifier(variable.getIdentifier());
    targetBatchingConstraint = BatchingConstraint(index.getValue(), targetID);
  }

  /// Optimize the value of the assignment
  auto cv = batchExpression(elem.getValue(), targetBatchingConstraint);

  /// Combine the execution plans, if they already exist
  if (variableValues.has(targetID)) {
    variableValues.getToModify(targetID).merge(cv);
  } else {
    precomputedValues.push_back(cv);
    variableValues.add(targetID, precomputedValues[precomputedValues.size() - 1]);
  }

  // Now delete this assignment
  delete_flag = true;
}

std::string SpecialVectorizer::getAuxiliaryInformation() {
  //TODO: Implement returning of auxiliary information
  return "NOT IMPLEMENTED YET";
}

bool isTransparentNode(const AbstractNode &node) {
  // a node is considered as transparent if it is an OperatorExpr because it can be batched by expanding any other
  // expression using the neutral element e.g., a and b*2 –- can be batched as a*1 and b*2
  return dynamic_cast<const BinaryExpression *>(&node)!=nullptr
      || dynamic_cast<const UnaryExpression *>(&node)!=nullptr
      || dynamic_cast<const OperatorExpression *>(&node)!=nullptr;
}

bool isBatchingCompatible(AbstractNode &baseNode, AbstractNode &curNode) {
  if (typeid(baseNode)!=typeid(curNode)) {
    // return true if...
    // - exactly one of both is transparent:
    //   (A XOR B)
    //   <=> (A && !B) || (!A && B)
    //   <=> (!A != !B)
    // - one of both is a AbstractLiteral
    return (!isTransparentNode(baseNode)!=!isTransparentNode(curNode))
        || isLiteral(baseNode)
        || isLiteral(curNode);
  } else {  // baseNode.type == curNode.type
    // type-specific checks
    if (auto baseNodeAsMatrixElementRef = dynamic_cast<IndexAccess *>(&baseNode)) {
      auto baseNodeVar = dynamic_cast<Variable *>(&baseNodeAsMatrixElementRef->getTarget());
      // as baseNode's type equals curNode's type, we know that curNodeAsMatrixElementRef != nullptr
      auto curNodeAsMatrixElementRef = dynamic_cast<IndexAccess *>(&curNode);
      auto curNodeVar = dynamic_cast<Variable *>(&curNodeAsMatrixElementRef->getTarget());
      if (baseNodeVar==nullptr || curNodeVar==nullptr) {
        throw std::runtime_error("IndexAccess unexpectedly does not refer to a Variable");
      }
      // check that both MatrixElementRefs refer to the same variable
      return baseNodeVar->getIdentifier()==curNodeVar->getIdentifier();
    } else if (auto baseNodeAsOperatorExpr = dynamic_cast<OperatorExpression *>(&baseNode)) {
      auto curNodeAsOperatorExpr = dynamic_cast<OperatorExpression *>(&curNode);
      // same operator
      return baseNodeAsOperatorExpr->getOperator()==curNodeAsOperatorExpr->getOperator()
          // same number of operands
          && baseNodeAsOperatorExpr->getOperands().size()==curNodeAsOperatorExpr->getOperands().size();
    } else { //TODO: Handle BinaryExpression and UnaryExpression!
      // handles all types that do not require any special handling, e.g., LiteralInt, Variable
      // (it is sufficient for batching compatibility that baseNode and curNode have the same type in that case)
      return true;
    }
  }
}

ComplexValue SpecialVectorizer::batchExpression(AbstractExpression &expr, BatchingConstraint) {

//  std::queue<std::reference_wrapper<AbstractExpression>> processingQ({expr});
//  while (!processingQ.empty()) {
//    auto &curNode = processingQ.front().get();
//    processingQ.pop();
//    std::cout << "Checking subtree rooted at " << curNode.getUniqueNodeId()
//              << " (#children: " << curNode.countChildren() << ")" << std::endl;
//    // returns the largest batchable subtree that was found
//    if (isBatchableSubtree(curNode)) return ComplexValue(curNode);
//    // otherwise we need to continue our search by checking all OperatorExprs on the next level
//    for (auto c : curNode->getChildrenNonNull()) { if (dynamic_cast<OperatorExpr *>(c)) processingQ.push(c); }
//  }
//  // nullptr = no batchable subtree found
//  return nullptr;


  std::vector<std::reference_wrapper<AbstractNode>> unprocessedNodes({expr});

  int numChildrenPerNode = -1;

  while (!unprocessedNodes.empty()) {

    // Get the next node to process
    auto &curNode = unprocessedNodes.front().get();
    unprocessedNodes.erase(unprocessedNodes.begin());

    // enqueue children to process next
    std::vector<std::reference_wrapper<AbstractNode>> childNodes;
    childNodes.insert(childNodes.end(), curNode.begin(), curNode.end());

    // Determine
    if (numChildrenPerNode==-1) {
      numChildrenPerNode = (int) curNode.countChildren();
    } else if (numChildrenPerNode!=curNode.countChildren()) {
      // a subtree is batching incompatible if the children have a different number of operands
      //TODO: Implement non-symmetric batching
      throw std::runtime_error("Batching of expressions with different number of children is not yet supported.");
    }


    // procedure that compares batching compatibility of nodes; if there are no more nodes that need to be processed
    // (unprocessedNodes.empty()) and there is no next level that needs to be checked (qWriting.empty()) then there is
    // nothing to check and this subtree is considered as batching-compatible
    if (unprocessedNodes.empty()) {

      // Check that all children are batching compatible to each other
      AbstractNode &baseNode = childNodes.front().get();
      for (auto nodeIt = std::next(childNodes.begin()); nodeIt!=childNodes.end(); ++nodeIt) {
        // check batching compatibility
        // TODO: Change this to actually construct the new expression as we go!
        if (!isBatchingCompatible(baseNode, *nodeIt)) {
          // if we detected a batching incompatibility, we can stop any further testing
          // TODO: Implement Support for this case!
          throw std::runtime_error("Rewriting of batching-incompatible expressions is not yet supported.");
        } else if (isTransparentNode(*nodeIt)) {
          // as we allow max. 1 transparent node per level, we need to make sure to compare any further transparent
          // nodes with the one we found here
          baseNode = *nodeIt;
        }
      } // end: for

      // move elements from childNodes to unprocessedNodes: childNodes is empty afterwards
      unprocessedNodes = std::move(childNodes);

      // reset #children counter back to default value
      numChildrenPerNode = -1;
    }
  } // end: while

  // if we processed all nodes and did not abort in between due to failed batching compatibility, the node rooted
  // at subtreeRoot is considered as batchable
  //bool isBatchable = qReading.empty() && qWriting.empty();

  //TODO: IMPLEMENT
  return ComplexValue(expr);
}