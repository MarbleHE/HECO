#include <memory>
#include <utility>
#include <iostream>

#include "ast_opt/visitor/runtime/RuntimeVisitor.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/runtime/AbstractCiphertextFactory.h"
#include "ast_opt/parser/Tokens.h"

template<typename S, typename T>
std::unique_ptr<T> castUniquePtr(std::unique_ptr<S> &&source) {
  if (dynamic_cast<T *>(source.get())) {
    return std::unique_ptr<T>(dynamic_cast<T *>(source.release()));
  } else {
    throw std::runtime_error("castUniquePtr failed: Cannot cast given unique_ptr from type "
                                 + std::string(typeid(S).name()) + " to type " + std::string(typeid(T).name()) + ".");
  }
}

std::unique_ptr<AbstractValue> SpecialRuntimeVisitor::getNextStackElement() {
  auto elem = std::move(intermedResult.top());
  intermedResult.pop();
  return elem;
}

void SpecialRuntimeVisitor::visit(BinaryExpression &elem) {
  // ---- some helper methods -------------------------
  auto operatorEqualsAnyOf = [&elem](std::initializer_list<OperatorVariant> op) -> bool {
    return std::any_of(op.begin(), op.end(), [&elem](OperatorVariant op) { return elem.getOperator()==Operator(op); });
  };
  auto operatorEquals = [&elem](OperatorVariant op) -> bool { return elem.getOperator()==Operator(op); };
  auto isSecretTainted = [&](const std::string &uniqueNodeId) -> bool {
    // we assume here that if it is NOT in the map, then it is NOT secret tainted
    return (secretTaintedMap.count(uniqueNodeId) > 0 && secretTaintedMap.at(uniqueNodeId));
  };
  // ---- end

  // if lhs or rhs are secret tainted but the operator is not FHE-compatible, throw an exception
  auto lhsIsSecret = isSecretTainted(elem.getLeft().getUniqueNodeId());
  auto rhsIsSecret = isSecretTainted(elem.getRight().getUniqueNodeId());
  if ((lhsIsSecret || rhsIsSecret) && !operatorEqualsAnyOf({FHE_ADDITION, FHE_SUBTRACTION, FHE_MULTIPLICATION})) {
    throw std::runtime_error("An operand in the binary expression is a ciphertext but given operation ("
                                 + elem.getOperator().toString() + ") cannot be executed on ciphertexts using FHE! "
                                 + "Note that you need to use FHE operators (+++, ---, ***) for ciphertext operations.\n"
                                 + "Expression: " + elem.toString(true));
  }

  elem.getLeft().accept(*this);
  auto lhsOperand = getNextStackElement();

  elem.getRight().accept(*this);
  auto rhsOperand = getNextStackElement();

  // if exactly one of the operands is a ciphertext and we have a commutative operation, then we make sure that
  // the first operand (the one we call the operation on) is the ciphertext
  if ((lhsIsSecret!=rhsIsSecret) && elem.getOperator().isCommutative()) {
    if (rhsIsSecret) std::swap(lhsOperand, rhsOperand);
  }

  // execute the binary operation
  if (operatorEqualsAnyOf({ADDITION, FHE_ADDITION})) {
    lhsOperand->add(*rhsOperand);
  } else if (operatorEqualsAnyOf({SUBTRACTION, FHE_SUBTRACTION})) {
    lhsOperand->subtract(*rhsOperand);
  } else if (operatorEqualsAnyOf({MULTIPLICATION, FHE_MULTIPLICATION})) {
    lhsOperand->multiply(*rhsOperand);
  } else if (operatorEquals(DIVISION)) {
    lhsOperand->divide(*rhsOperand);
  } else if (operatorEquals(MODULO)) {
    lhsOperand->modulo(*rhsOperand);
  } else if (operatorEquals(LOGICAL_AND)) {
    lhsOperand->logicalAnd(*rhsOperand);
  } else if (operatorEquals(LOGICAL_OR)) {
    lhsOperand->logicalOr(*rhsOperand);
  } else if (operatorEquals(LESS)) {
    lhsOperand->logicalLess(*rhsOperand);
  } else if (operatorEquals(LESS_EQUAL)) {
    lhsOperand->logicalLessEqual(*rhsOperand);
  } else if (operatorEquals(GREATER)) {
    lhsOperand->logicalGreater(*rhsOperand);
  } else if (operatorEquals(GREATER_EQUAL)) {
    lhsOperand->logicalGreaterEqual(*rhsOperand);
  } else if (operatorEquals(EQUAL)) {
    lhsOperand->logicalEqual(*rhsOperand);
  } else if (operatorEquals(NOTEQUAL)) {
    lhsOperand->logicalNotEqual(*rhsOperand);
  } else if (operatorEquals(BITWISE_AND)) {
    lhsOperand->bitwiseAnd(*rhsOperand);
  } else if (operatorEquals(BITWISE_XOR)) {
    lhsOperand->bitwiseXor(*rhsOperand);
  } else if (operatorEquals(BITWISE_OR)) {
    lhsOperand->bitwiseOr(*rhsOperand);
  } else {
    throw std::runtime_error("Unknown binary operator encountered. Cannot continue!");
  }
  intermedResult.push(std::move(lhsOperand));
}

void SpecialRuntimeVisitor::visit(UnaryExpression &elem) {
  elem.getOperand().accept(*this);
  auto operand = getNextStackElement();

  if (elem.getOperator()==Operator(LOGICAL_NOT)) {
    operand->logicalNot();
  } else if (elem.getOperator()==Operator(BITWISE_NOT)) {
    operand->bitwiseNot();
  } else {
    throw std::runtime_error("Unknown unary operator encountered!");
  }
}

void SpecialRuntimeVisitor::visit(Call &elem) {
  if (elem.getIdentifier()!=stork::to_string(stork::reservedTokens::kw_rotate)) {
    throw std::runtime_error("Calls other than 'rotate(identifier: label, numSteps: int);' are not supported yet!");
  }

  // handle 'rotate' instruction
  if (elem.getArguments().size() < 2) {
    throw std::runtime_error(
        "Instruction 'rotate' requires two arguments: (1) identifier of ciphertext to be rotated "
        "and the (2) number of steps to rotate the ciphertext.");
  }

  // arg 0: ciphertext to rotate
  auto ciphertextIdentifier = elem.getArguments().at(0);
  std::unique_ptr<AbstractCiphertext> ctxt;
  auto ciphertextIdentifierVariable = dynamic_cast<Variable *>(&ciphertextIdentifier.get());
  if (ciphertextIdentifierVariable==nullptr) {
    throw std::runtime_error("Argument 'ciphertext' in 'rotate' instruction must be a variable.");
  }
  auto scopedIdentifier = getCurrentScope().resolveIdentifier(ciphertextIdentifierVariable->getIdentifier());

  // arg 1: rotation steps
  auto steps = elem.getArguments().at(1);
  auto stepsLiteralInt = dynamic_cast<LiteralInt *>(&steps.get());
  if (stepsLiteralInt==nullptr) {
    throw std::runtime_error("Argument 'steps' in 'rotate' instruction must be an integer.");
  }

  // perform rotation
  auto rotatedCtxt = declaredCiphertexts.at(scopedIdentifier)->rotateRows(stepsLiteralInt->getValue());
  intermedResult.push(std::move(rotatedCtxt));
}

void SpecialRuntimeVisitor::visit(ExpressionList &elem) {
  ScopedVisitor::visit(elem);
  // after visiting the expression list with this visitor, there should only be Literals left, otherwise this expression
  // list is not valid; we use here that the TypeCheckingVisitor verified that all expressions in an ExpressionList have
  // the same type

  std::vector<std::unique_ptr<ICleartext>> cleartextVec;
  for (size_t i = 0; i < elem.getExpressions().size(); ++i) {
    auto e = getNextStackElement();
    if (dynamic_cast<ICleartext *>(e.get())) {
      std::unique_ptr<ICleartext> derivedPointer(dynamic_cast<ICleartext *>(e.release()));
      // We are now processing elements of the ExpressionList in reverse order, i.e., the last visited element of the
      // ExpressionList is on the top of the stack. Thus, we need to append elements in the cleartextVec to the front.
      cleartextVec.insert(cleartextVec.begin(), std::move(derivedPointer));
    } else {
      throw std::runtime_error("Found ExpressionList that does contain any other than ICleartext element. Aborting...");
    }
  }

  auto firstExpression = elem.getExpressions().at(0);
  if (dynamic_cast<LiteralBool *>(&firstExpression.get())) {
    intermedResult.emplace(std::make_unique<Cleartext<LiteralBool::value_type>>(cleartextVec));
  } else if (dynamic_cast<LiteralChar *>(&elem)) {
    intermedResult.emplace(std::make_unique<Cleartext<LiteralChar::value_type>>(cleartextVec));
  } else if (dynamic_cast<LiteralInt *>(&firstExpression.get())) {
    intermedResult.emplace(std::make_unique<Cleartext<LiteralInt::value_type>>(cleartextVec));
  } else if (dynamic_cast<LiteralFloat *>(&firstExpression.get())) {
    intermedResult.emplace(std::make_unique<Cleartext<LiteralFloat::value_type>>(cleartextVec));
  } else if (dynamic_cast<LiteralDouble *>(&firstExpression.get())) {
    intermedResult.emplace(std::make_unique<Cleartext<LiteralDouble::value_type>>(cleartextVec));
  } else if (dynamic_cast<LiteralString *>(&firstExpression.get())) {
    intermedResult.emplace(std::make_unique<Cleartext<LiteralString::value_type>>(cleartextVec));
  } else {
    throw std::runtime_error("Could not determine element type of ExpressionList!");
  }
}

void SpecialRuntimeVisitor::visit(For &elem) {
  elem.getInitializer().accept(*this);

  // a helper method to check the value of the For loop's condition
  auto evaluateCondition = [&](AbstractExpression &expr) -> bool {
    expr.accept(*this);
    auto result = getNextStackElement();
    if (auto conditionLiteralBool = dynamic_cast<LiteralBool *>(result.get())) {
      return conditionLiteralBool->getValue();
    } else {
      throw std::runtime_error("For loop's condition must be evaluable to a Boolean.");
    }
  };

  if (elem.hasCondition() && secretTaintedMap.count(elem.getCondition().getUniqueNodeId()) > 0) {
    throw std::runtime_error("For loops over secret conditions are not supported yet!");
  }

  // execute the For loop
  if (elem.hasCondition()) {
    while (evaluateCondition(elem.getCondition())) {
      if (elem.hasBody()) elem.getBody().accept(*this);
      if (elem.hasUpdate()) elem.getUpdate().accept(*this);
    }
  } else {
    throw std::runtime_error("For loops without a condition are not supported yet!");
  }
}

void SpecialRuntimeVisitor::visit(Function &) {
  throw std::runtime_error("Function statements are not supported yet by RuntimeVisitor.");
}

void SpecialRuntimeVisitor::visit(If &elem) {
  // check if the If statement's condition is secret
  // (although we ran the SecretBranchingVisitor before, it can be that there are still secret conditions left,
  // for example, if the then/else branch contains an unsupported statement such as a loop)
  if (secretTaintedMap.at(elem.getCondition().getUniqueNodeId())) {
    throw std::runtime_error("If statements over secret conditions that cannot be rewritten using the "
                             "SecretBranchingVisitor are not supported yet!");
  }

  // get the If statement's condition
  elem.getCondition().accept(*this);
  auto conditionResult = getNextStackElement();

  if (auto conditionLiteralBool = dynamic_cast<Cleartext<bool> *>(conditionResult.get())) {
    if (conditionLiteralBool->getData().front()) {
      // visit "then" branch
      elem.getThenBranch().accept(*this);
    } else if (elem.hasElseBranch()) {
      // visit "else" branch if existent
      elem.getElseBranch().accept(*this);
    }
  } else {
    throw std::runtime_error("Condition of If statement must be evaluable to a bool.");
  }
}

void SpecialRuntimeVisitor::visit(IndexAccess &elem) {
  if (secretTaintedMap.at(elem.getUniqueNodeId())) {
    throw std::runtime_error("IndexAccess for secret variables is not supported by RuntimeVisitor. "
                             "This should have already been removed by the Vectorizer. Error?");
  }

  elem.getTarget().accept(*this);
  auto target = getNextStackElement();

  elem.getIndex().accept(*this);
  auto index = getNextStackElement();

  // we need to cast the target and the index of this IndexAccess to perform the action
  auto castedCleartext = dynamic_cast<Cleartext<int> *>(target.get());
  auto castedIndex = dynamic_cast<Cleartext<int> *>(index.get());
  if (castedCleartext==nullptr || castedIndex==nullptr) {
    throw std::runtime_error("IndexAccess only implemented for Cleartext<int> yet.");
  }

  if (!castedIndex->allEqual()) {
    throw std::runtime_error("The resolved index of the IndexAccess doesn't seem like to be a scalar integer.");
  }

  // we create a new Cleartext<int> that only contains the referenced value
  auto singleValue = castedCleartext->getData().at(castedIndex->getData().at(0));
  std::unique_ptr<Cleartext<int>> newCleartext = std::make_unique<Cleartext<int>>(std::vector<int>({singleValue}));

  intermedResult.push(std::move(newCleartext));
}

void SpecialRuntimeVisitor::visit(LiteralBool &elem) {
  visitHelper(elem);
}

void SpecialRuntimeVisitor::visit(LiteralChar &elem) {
  visitHelper(elem);
}

void SpecialRuntimeVisitor::visit(LiteralInt &elem) {
  visitHelper(elem);
}

void SpecialRuntimeVisitor::visit(LiteralFloat &elem) {
  visitHelper(elem);
}

void SpecialRuntimeVisitor::visit(LiteralDouble &elem) {
  visitHelper(elem);
}

void SpecialRuntimeVisitor::visit(LiteralString &elem) {
  visitHelper(elem);
}

void SpecialRuntimeVisitor::visit(OperatorExpression &) {
  throw std::runtime_error("UNIMPLEMENTED: RuntimeVisitor cannot execute OperatorExpressions yet.");
}

void SpecialRuntimeVisitor::visit(Return &elem) {
  ScopedVisitor::visit(elem);
  throw ReturnStatementReached();
}

void SpecialRuntimeVisitor::visit(Assignment &elem) {
  elem.getTarget().accept(*this);
  auto assignmentTarget = getNextStackElement();

  elem.getValue().accept(*this);
  auto assignmentValue = getNextStackElement();

  auto isSecretVariable = [&](const ScopedIdentifier &scopedIdentifier) {
    return identifierDatatypes.at(scopedIdentifier).getSecretFlag();
  };

  auto atVariable = dynamic_cast<Variable *>(&elem.getTarget());
  auto atCiphertext = dynamic_cast<AbstractCiphertext *>(assignmentTarget.get());

  if (atVariable!=nullptr && atCiphertext!=nullptr) {
    auto scopedIdentifier = getCurrentScope().resolveIdentifier(atVariable->getIdentifier());
    std::unique_ptr<AbstractCiphertext>
        actxt = castUniquePtr<AbstractValue, AbstractCiphertext>(std::move(assignmentValue));
    declaredCiphertexts.insert_or_assign(scopedIdentifier, std::move(actxt));
  } else if (atVariable!=nullptr) {
    auto scopedIdentifier = getCurrentScope().resolveIdentifier(atVariable->getIdentifier());
    // check if this assignment targets a secret variable
    if (isSecretVariable(scopedIdentifier)) {
      // we need to convert the std::unique_ptr<AbstractValue> into a std::unique_ptr<AbstractCiphertext>
      auto ciphertext = castUniquePtr<AbstractValue, AbstractCiphertext>(std::move(assignmentValue));
      declaredCiphertexts.insert_or_assign(scopedIdentifier, std::move(ciphertext));
    } else {
      auto cleartext = castUniquePtr<AbstractValue, ICleartext>(std::move(assignmentValue));
      declaredCleartexts.insert_or_assign(scopedIdentifier, std::move(cleartext));
    }
  } else if (auto atAssignm = dynamic_cast<IndexAccess *>(&elem.getTarget())) {
    if (auto var = dynamic_cast<Variable *>(&atAssignm->getTarget())) {
      // retrieve the index of this IndexAccess
      atAssignm->getIndex().accept(*this);
      auto idx = getNextStackElement();
      auto idxAsInt = dynamic_cast<Cleartext<int> *>(idx.get());
      if (idxAsInt==nullptr) {
        throw std::runtime_error("Index given in IndexAccess must be an integer!");
      } else if (!idxAsInt->allEqual()) {
        throw std::runtime_error("Index of IndexAccess must be a scalar.");
      }
      // now update the cleartext at the determined index with the given value
      auto scopedIdentifier = getCurrentScope().resolveIdentifier(var->getIdentifier());
      declaredCleartexts.at(scopedIdentifier)->setValueAtIndex(idxAsInt->getData().at(0), std::move(assignmentValue));
    } else {
      throw std::runtime_error(
          "Only simple, non-nested IndexAccesses on non-secret variables are supported yet "
          "(e.g., i[2] -> ok, i[j[2]] -> not supported).");
    }
  } else {
    throw std::runtime_error("Assignments currently only supported to (non-indexed) variables.");
  }
}

void SpecialRuntimeVisitor::visit(VariableDeclaration &elem) {
  auto scopedIdentifier = std::make_unique<ScopedIdentifier>(getCurrentScope(), elem.getTarget().getIdentifier());
  identifierDatatypes.emplace(*scopedIdentifier, elem.getDatatype());
  getCurrentScope().addIdentifier(std::move(scopedIdentifier));

  // if this declaration does not have an initialization, we can stop here as there's no value we need to keep track of
  if (!elem.hasValue()) return;

  // after having visited the variable declaration's initialization value, then we should have a Cleartext<T> on the
  // top of the intermedResult stack
  elem.getValue().accept(*this);
  std::unique_ptr<AbstractValue> initializationValue = getNextStackElement();

  if (elem.getDatatype().getSecretFlag()) {
    auto sident = std::make_unique<ScopedIdentifier>(getCurrentScope(), elem.getTarget().getIdentifier());
    // declaration of a secret variable: we need to check whether we need to create a ciphertext here or whether the
    // right-hand side of the expression already created a ciphertext for us (if involved operands were ciphertexts too)
    if (dynamic_cast<AbstractCiphertext *>(initializationValue.get())) {  // use the result ciphertext from RHS
      auto ctxt = castUniquePtr<AbstractValue, AbstractCiphertext>(std::move(initializationValue));
      declaredCiphertexts.insert_or_assign(*sident, std::move(ctxt));
    } else {  // create a new ciphertext
      auto ctxt = factory.createCiphertext(std::move(initializationValue));
      declaredCiphertexts.insert_or_assign(*sident, std::move(ctxt));
    }
  } else if (dynamic_cast<ICleartext *>(initializationValue.get())) {
    // declaration of a non-secret variable
    // we need to convert std::unique_ptr<AbstractValue> from intermedResult into std::unique_ptr<ICleartext>
    // as we now that this variable declaration's value is not secret tainted, this must be a cleartext
    auto sident = std::make_unique<ScopedIdentifier>(getCurrentScope(), elem.getTarget().getIdentifier());
    auto cleartextUniquePtr = castUniquePtr<AbstractValue, ICleartext>(std::move(initializationValue));
    declaredCleartexts.insert_or_assign(*sident, std::move(cleartextUniquePtr));
  } else {
    throw std::runtime_error("Initialization value of VariableDeclaration ( " + elem.getTarget().getIdentifier()
                                 + ") could not be processed successfully.");
  }
}

void SpecialRuntimeVisitor::visit(Variable &elem) {
  auto scopedIdentifier = getCurrentScope().resolveIdentifier(elem.getIdentifier());
  // in both cases we need to clone the underlying type (AbstractCiphertext or Cleartext) as the maps
  // (declaredCiphertexts and declaredCleartexts) holds ownership and it could be that the same variable will be
  // referenced later again
  if (identifierDatatypes.at(scopedIdentifier).getSecretFlag()) {
    // variable refers to an encrypted value, i.e., is a ciphertext
    auto clonedCiphertext = declaredCiphertexts.at(scopedIdentifier)->clone();
    intermedResult.emplace(std::move(clonedCiphertext));
  } else {
    // variable refers to a cleartext value
    auto clonedCleartext = declaredCleartexts.at(scopedIdentifier)->clone();
    intermedResult.emplace(std::move(clonedCleartext));
  }
}

template<typename T>
void SpecialRuntimeVisitor::checkAstStructure(AbstractNode &astRootNode) {
  /// The input and output ASTs are expected to consist of a single block with variable declaration
  /// statements and variable assignments, respectively.

  if (dynamic_cast<Block *>(&astRootNode)==nullptr)
    throw std::runtime_error("Root of (in-/out)put AST must be a Block node.");

  // check each statement of the AST
  for (auto &statement : astRootNode) {
    // check that statements of Block are of expected type
    auto castedStatement = dynamic_cast<T *>(&statement);
    if (castedStatement==nullptr) {
      throw std::runtime_error("Block statements of given (in-/out)put AST must be of type "
                                   + std::string(typeid(T).name()) + ". ");
    }

    // special condition for assignments: require that assignment's value is a Variable or IndexAccess
    if (typeid(T)==typeid(Assignment)) {
      auto valueAsVariable = dynamic_cast<Variable *>(&castedStatement->getTarget());
      auto valueAsIndexAccess = dynamic_cast<IndexAccess *>(&castedStatement->getTarget());
      if (valueAsVariable==nullptr && valueAsIndexAccess==nullptr) {
        throw std::runtime_error("Output AST must consist of Assignments to variables, i.e., Variable or IndexAccess.");
      }
    }
  }
}

SpecialRuntimeVisitor::SpecialRuntimeVisitor(
    AbstractCiphertextFactory &factory, AbstractNode &inputs, SecretTaintedNodesMap &secretTaintedNodesMap)
    : factory(factory), secretTaintedMap(secretTaintedNodesMap) {
  // generate ciphertexts for inputs
  checkAstStructure<VariableDeclaration>(inputs);
  inputs.accept(*this);
}

void SpecialRuntimeVisitor::executeAst(AbstractNode &rootNode) {
  try {
    rootNode.accept(*this);
  } catch (ReturnStatementReached &) {
    std::cout << "Program reached return statement.." << std::endl;
  }
}

OutputIdentifierValuePairs SpecialRuntimeVisitor::getOutput(AbstractNode &outputAst) {
  // make sure that outputAst consists of a Block with Assignment statements
  checkAstStructure<Assignment>(outputAst);

  // extract lhs and rhs of assignment
  OutputIdentifierValuePairs outputValues;
  auto block = dynamic_cast<Block &>(outputAst);
  for (auto &assignm : block.getStatements()) {
    // extract assignment's target (lhs)
    auto varAssignm = dynamic_cast<Assignment &>(assignm.get());
    auto identifier = dynamic_cast<Variable &>(varAssignm.getTarget()).getIdentifier();

    // extract assignment's value (rhs): either a Variable or an IndexAccess
    std::unique_ptr<AbstractValue> result;
    if (auto valueAsVariable = dynamic_cast<Variable *>(&varAssignm.getValue())) {
      // if the value is a Variable: it's sufficient if we clone the corresponding ciphertext
      auto scopedIdentifier = getCurrentScope().resolveIdentifier(valueAsVariable->getIdentifier());
      if (identifierDatatypes.at(scopedIdentifier).getSecretFlag()) {
        result = declaredCiphertexts.at(scopedIdentifier)->clone();
      } else {
        result = declaredCleartexts.at(scopedIdentifier)->clone();
      }
    } else if (auto valueAsIndexAccess = dynamic_cast<IndexAccess *>(&varAssignm.getValue())) {
      // if the value is an IndexAccess we need to clone & rotate the ciphertext accordingly
      try {
        auto valueIdentifier = dynamic_cast<Variable &>(valueAsIndexAccess->getTarget());
        auto idx = dynamic_cast<LiteralInt &>(valueAsIndexAccess->getIndex());
        auto scopedIdentifier = getRootScope().resolveIdentifier(valueIdentifier.getIdentifier());
        result = declaredCiphertexts.at(scopedIdentifier)->rotateRows(idx.getValue());
      } catch (std::bad_cast &) {
        throw std::runtime_error(
            "Nested index accesses in right-hand side of output AST not allowed (e.g., y = __input0__[a[2]]).");
      }
    } else {
      throw std::runtime_error("Right-hand side of output AST is neither a Variable nor IndexAccess "
                               "(e.g., y = __input0__ or y = __input0__[2]).");
    }
    outputValues.emplace_back(identifier, std::move(result));
  } // end: for (auto &assignm : block.getStatements())

  return outputValues;
}

[[maybe_unused]] void SpecialRuntimeVisitor::printOutput(AbstractNode &outputAst, std::ostream &targetStream) {
  // retrieve the identifiers mentioned in the output AST, decrypt referred ciphertexts, and print them
  auto outputValues = getOutput(outputAst);
  for (const auto &v : outputValues) {
    targetStream << v.first << ": ";
    if (auto vAsAbstractCiphertext = dynamic_cast<AbstractCiphertext *>(v.second.get())) {
      targetStream << factory.getString(*vAsAbstractCiphertext) << std::endl;
    } else if (auto vAsCleartext = dynamic_cast<ICleartext *>(v.second.get())) {
      targetStream << vAsCleartext->toString() << std::endl;
    }
  }
}
