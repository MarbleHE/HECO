#include <utility>
#include <iostream>

#include "ast_opt/visitor/Runtime/RuntimeVisitor.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/Runtime/AbstractCiphertextFactory.h"
#include "ast_opt/parser/Tokens.h"

AbstractExpression &SpecialRuntimeVisitor::getNextStackElement() {
  auto elem = intermedResult.top();
  intermedResult.pop();
  return elem.get();
}

void SpecialRuntimeVisitor::visit(BinaryExpression &elem) {
  auto operatorEqualsAnyOf = [&elem](std::initializer_list<OperatorVariant> op) -> bool {
    return std::any_of(op.begin(), op.end(),
                       [&elem](OperatorVariant op) { return elem.getOperator()==Operator(op); });
  };

  // TODO: if lhs or rhs are secret tainted but the operator is non-FHE compatible, throw an exception
  if ((secretTaintedMap.at(elem.getLeft().getUniqueNodeId()) || secretTaintedMap.at(elem.getRight().getUniqueNodeId()))
      && !operatorEqualsAnyOf({FHE_ADDITION, FHE_SUBTRACTION, FHE_MULTIPLICATION})) {
    throw std::runtime_error("An operand in the binary expression is a ciphertext but given operation ("
                                 + elem.getOperator().toString() + ") cannot be executed on ciphertexts!\n"
                                 + "Expression: " + elem.toString(false));
  }

  elem.getLeft().accept(*this);
  // auto &lhsOperand = getNextStackElement();

  elem.getRight().accept(*this);
  // auto &rhsOperand = getNextStackElement();

  auto operatorEquals = [&elem](OperatorVariant op) -> bool { return elem.getOperator()==Operator(op); };



  // TODO: Implement me!
  if (operatorEqualsAnyOf({ADDITION, FHE_ADDITION})) {

  } else if (operatorEqualsAnyOf({SUBTRACTION, FHE_SUBTRACTION})) {

  } else if (operatorEqualsAnyOf({MULTIPLICATION, FHE_MULTIPLICATION})) {

  } else if (elem.getOperator()==Operator(DIVISION)) {

  } else if (elem.getOperator()==Operator(MODULO)) {

  } else if (elem.getOperator()==Operator(LOGICAL_AND)) {

  } else if (elem.getOperator()==Operator(LOGICAL_OR)) {

  } else if (elem.getOperator()==Operator(LESS)) {

  } else if (elem.getOperator()==Operator(LESS_EQUAL)) {

  } else if (elem.getOperator()==Operator(GREATER)) {

  } else if (elem.getOperator()==Operator(GREATER_EQUAL)) {

  } else if (elem.getOperator()==Operator(EQUAL)) {

  } else if (elem.getOperator()==Operator(NOTEQUAL)) {

  } else if (elem.getOperator()==Operator(BITWISE_AND)) {

  } else if (elem.getOperator()==Operator(BITWISE_XOR)) {

  } else if (elem.getOperator()==Operator(BITWISE_OR)) {

  }
}

void SpecialRuntimeVisitor::visit(UnaryExpression &elem) {
  ScopedVisitor::visit(elem);



  // TODO: Implement me!
  if (elem.getOperator()==Operator(LOGICAL_NOT)) {

  } else if (elem.getOperator()==Operator(BITWISE_NOT)) {

  } else {
    throw std::runtime_error("Unknown unary operator encountered!");
  }
}

void SpecialRuntimeVisitor::visit(Block &elem) {
  ScopedVisitor::visit(elem);
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
  ciphertexts.at(scopedIdentifier)->rotateRowsInplace(stepsLiteralInt->getValue());
}

void SpecialRuntimeVisitor::visit(ExpressionList &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(For &elem) {
  elem.getInitializer().accept(*this);

  // a helper method to check the value of the For loop's condition
  auto evaluateCondition = [&](AbstractExpression &expr) -> bool {
    expr.accept(*this);
    auto &result = getNextStackElement();
    if (auto conditionLiteralBool = dynamic_cast<LiteralBool *>(&result)) {
      return conditionLiteralBool->getValue();
    } else {
      throw std::runtime_error("For loop's condition must be evaluable to a Boolean.");
    }
  };

  // TODO: Do we support loop's over a secret condition? If not, include check in elem.hasCondition() to make sure that
  //  the condition is not secret -> otherwise throw exception

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
  ScopedVisitor::visit(elem);

  // check if the If statement's condition is secret
  // (although we ran the SecretBranchingVisitor before, it can be that there are still secret conditions left,
  // for example, if the then/else branch contains an unsupported statement such as a loop)



  // get the If statement's condition
  elem.getCondition().accept(*this);
  auto &conditionResult = getNextStackElement();

  if (auto conditionLiteralBool = dynamic_cast<LiteralBool *>(&conditionResult)) {
    if (conditionLiteralBool->getValue()) {
      // visit "then" branch
      elem.getThenBranch().accept(*this);
    } else if (elem.hasElseBranch()) {
      // visit "else" branch if existent
      elem.getElseBranch().accept(*this);
    }
  } else {
    throw std::runtime_error("Condition of If statement must evaluate to a LiteralBool");
  }
}

void SpecialRuntimeVisitor::visit(IndexAccess &elem) {
  ScopedVisitor::visit(elem);
  // TODO: Implement me!

  elem.getTarget().accept(*this);

  elem.getIndex().accept(*this);
}

void SpecialRuntimeVisitor::visit(LiteralBool &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralChar &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralInt &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralFloat &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralDouble &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralString &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(OperatorExpression &) {
  throw std::runtime_error("UNIMPLEMENTED: RuntimeVisitor cannot execute OperatorExpressions yet.");
}

void SpecialRuntimeVisitor::visit(Return &elem) {
  ScopedVisitor::visit(elem);
  // TODO: Implement me!
}

void SpecialRuntimeVisitor::visit(Assignment &elem) {

  elem.getTarget().accept(*this);
  // auto &assignmentTarget = getNextStackElement();

//  //
//  if (auto atVariable = dynamic_cast<Variable *>(&assignmentTarget)) {
//    assignmentTargetIdentifiers.emplace_back(getCurrentScope(), atVariable->getIdentifier());
//  } else if (auto atIndexAccess = dynamic_cast<IndexAccess *>(&assignmentTarget)) {
//    // after visiting the target, the index access should not contain any nested index accesses anymore,
//    // for example, i[k[2]] -> i[4] assuming k[2] = 4
//    if (auto indexAccessVariable = dynamic_cast<Variable *>(&atIndexAccess->getTarget())) {
//      assignmentTargetIdentifiers.emplace_back(getCurrentScope(), indexAccessVariable->getIdentifier());
//    } else {
//      throw std::runtime_error("");
//    }
//  }

  elem.getValue().accept(*this);


  // TODO: Implement me!
}

std::vector<int64_t> extractIntegerFromLiteralInt(ExpressionList &el) {
  std::vector<int64_t> result;
  for (std::reference_wrapper<AbstractExpression> &abstractExpr : el.getExpressions()) {
    if (auto casted = dynamic_cast<LiteralInt *>(&abstractExpr.get())) {
      result.push_back(casted->getValue());
    } else {
      throw std::runtime_error("Elements of ExpressionList are expected to be LiteralInts, "
                               "found " + std::string(typeid(abstractExpr).name()) + " instead.");
    }
  }
  return result;
}

//template<typename T>
//T SpecialRuntimeVisitor::getValue(ScopedIdentifier &scopedIdentifier) {
//  const auto identifierExists = identifierDatatypes.count(scopedIdentifier)!=0;
//  if (identifierExists && identifierDatatypes.at(scopedIdentifier).getSecretFlag()==true) {
//    return ciphertexts.at(scopedIdentifier);
//  } else if (identifierExists && identifierDatatypes.at(scopedIdentifier).getSecretFlag()==false) {
//    return 0;
//  } else {
//    throw std::runtime_error("");
//  };
//}

void SpecialRuntimeVisitor::visit(VariableDeclaration &elem) {
  auto scopedIdentifier = std::make_unique<ScopedIdentifier>(getCurrentScope(), elem.getTarget().getIdentifier());
  identifierDatatypes.emplace(*scopedIdentifier,
                              elem.getDatatype());
  getCurrentScope().addIdentifier(std::move(scopedIdentifier));

  if (elem.hasValue()) elem.getValue().accept(*this);
  auto &initializationValue = getNextStackElement();

  // TODO: Implement me!
  if (elem.getDatatype().getSecretFlag()) {
    // declaration of a secret variable, i.e., a ciphertext
    if (auto ivAsLiteralInt = dynamic_cast<LiteralInt *>(&initializationValue)) {
      auto sident = std::make_unique<ScopedIdentifier>(getCurrentScope(), elem.getTarget().getIdentifier());
      auto ciphertext = factory.createCiphertext(ivAsLiteralInt->getValue());

      ciphertexts.insert_or_assign(*sident, std::move(ciphertext));
    } else if (auto ivAsExprList = dynamic_cast<ExpressionList *>(&initializationValue)) {
      auto sident = std::make_unique<ScopedIdentifier>(getCurrentScope(), elem.getTarget().getIdentifier());
      auto vec = extractIntegerFromLiteralInt(*ivAsExprList);
      auto ciphertext = factory.createCiphertext(vec);
      ciphertexts.insert_or_assign(*sident, std::move(ciphertext));
    } else {
      throw std::runtime_error("Cannot create a ciphertext for anything other than a LiteralInt or an ExpressionList "
                               "of LiteralInts yet.");
    }
  } else {
    // declaration of a non-secret variable


  }
}

void SpecialRuntimeVisitor::visit(Variable &elem) {
  // TODO: Implement me!
  auto scopedIdentifier = getCurrentScope().resolveIdentifier(elem.getIdentifier());
  if (identifierDatatypes.at(scopedIdentifier).getSecretFlag()) {
//    auto &value = ciphertexts.at(scopedIdentifier);
  } else {
//    auto value = plainValues.at(scopedIdentifier);
  }
}

template<typename T>
void SpecialRuntimeVisitor::checkAstStructure(AbstractNode &astRootNode) {
  /// The input and output ASTs are expected to consist of a single block with variable declaration
  /// statements and variable assignments, respectively.

  if (dynamic_cast<Block *>(&astRootNode)==nullptr) {
    throw std::runtime_error("Root node of input/output AST is expected to be a Block node.");
  }

  for (auto &statement : astRootNode) {
    auto castedStatement = dynamic_cast<T *>(&statement);
    // check that statements of Block are of expected type
    if (castedStatement==nullptr) {
      throw std::runtime_error(
          "Block statements of given (input|output) AST are expected to be of type "
              + std::string(typeid(T).name()) + ". ");
    }
    // special condition for assignments: require that assignment's value is a Variable
    if (typeid(T)==typeid(Assignment)) {
      auto valueAsVariable = dynamic_cast<Variable *>(&castedStatement->getTarget());
      auto valueAsIndexAccess = dynamic_cast<IndexAccess *>(&castedStatement->getTarget());
      if (valueAsVariable==nullptr && valueAsIndexAccess==nullptr) {
        throw std::runtime_error(
            "Output AST must consist of Assignments to (indexed) variables (i.e., Variable or IndexAccess).");
      }
    }
  }
}

SpecialRuntimeVisitor::SpecialRuntimeVisitor(AbstractCiphertextFactory &factory,
                                             AbstractNode &inputs,
                                             SecretTaintedNodesMap &secretTaintedNodesMap)
    : factory(factory), secretTaintedMap(secretTaintedNodesMap) {
  // generate ciphertexts for inputs
  checkAstStructure<VariableDeclaration>(inputs);
  inputs.accept(*this);
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
    std::unique_ptr<AbstractCiphertext> ctxt;
    if (auto valueAsVariable = dynamic_cast<Variable *>(&varAssignm.getValue())) {
      // if the value is a Variable, it's sufficient if we get the corresponding ciphertext
      auto scopedIdentifier = getRootScope().resolveIdentifier(valueAsVariable->getIdentifier());
      // TODO: This modifies the output s.t. calling the same method or printOutput afterward will not work.
      //  Create a copy-constructor in AbstractCiphertext and use copies instead. Then make this method const.
      ctxt = ciphertexts.at(scopedIdentifier)->clone();
    } else if (auto valueAsIndexAccess = dynamic_cast<IndexAccess *>(&varAssignm.getValue())) {
      // if the value is an IndexAccess
      try {
        auto valueIdentifier = dynamic_cast<Variable &>(valueAsIndexAccess->getTarget());
        auto idx = dynamic_cast<LiteralInt &>(valueAsIndexAccess->getIndex());
        auto scopedIdentifier = getRootScope().resolveIdentifier(valueIdentifier.getIdentifier());
        ctxt = ciphertexts.at(scopedIdentifier)->rotateRows(idx.getValue());
      } catch (std::bad_cast &) {
        throw std::runtime_error(
            "Nested index accesses in right-hand side of output AST not allowed (e.g., y = __input0__[a[2]]).");
      }
    } else {
      throw std::runtime_error("Right-hand side of output AST is neither a Variable nor IndexAccess "
                               "(e.g., y = __input0__ or y = __input0__[2]).");
    }
    outputValues.emplace_back(identifier, std::move(ctxt));
  } // end: for (auto &assignm : block.getStatements())

  return outputValues;
}

void SpecialRuntimeVisitor::printOutput(AbstractNode &outputAst, std::ostream &targetStream) {
  // retrieve the identifiers mentioned in the output AST, decrypt referred ciphertexts, and print them
  auto outputValues = getOutput(outputAst);
  for (const auto &v : outputValues) {
    targetStream << v.first << ": " << factory.getString(*v.second) << std::endl;
  }
}
