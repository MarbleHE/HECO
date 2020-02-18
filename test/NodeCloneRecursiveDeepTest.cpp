#include "gtest/gtest.h"
#include "AbstractExpr.h"
#include "AbstractStatement.h"
#include "BinaryExpr.h"
#include "Block.h"
#include "Call.h"
#include "CallExternal.h"
#include "Function.h"
#include "FunctionParameter.h"
#include "If.h"
#include "AbstractLiteral.h"
#include "LiteralBool.h"
#include "LiteralFloat.h"
#include "LiteralInt.h"
#include "LiteralString.h"
#include "LogicalExpr.h"
#include "OpSymbEnum.h"
#include "Operator.h"
#include "Return.h"
#include "UnaryExpr.h"
#include "VarAssignm.h"
#include "VarDecl.h"
#include "Variable.h"

class NodeCloneTestFixture : public ::testing::Test {
 protected:
  NodeCloneTestFixture() = default;

  static void assertNodeAttributes(bool keepOriginalUniqueId, AbstractNode *original, AbstractNode *clone) {
    if (keepOriginalUniqueId) {
      ASSERT_EQ(original->getUniqueNodeId(), clone->getUniqueNodeId());
    } else {
      ASSERT_NE(original->getUniqueNodeId(), clone->getUniqueNodeId());
    }
    ASSERT_EQ(original->isReversed, clone->isReversed);
  }
};

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_BinaryExpr) {  /* NOLINT */
  const bool keepOriginalId = true;

  // create a new binary expression
  auto lhsOperand = new LiteralInt(0);
  auto operatore = new Operator(OpSymb::addition);
  auto rhsOperand = new LiteralInt(987);
  auto binaryExpression = new BinaryExpr(lhsOperand, operatore, rhsOperand);
  // clone the logical expression
  auto clonedBinaryExprAsNode = binaryExpression->clone(keepOriginalId);
  auto clonedBinaryExprCasted = clonedBinaryExprAsNode->castTo<BinaryExpr>();
  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, binaryExpression, clonedBinaryExprAsNode);

  // make changes to the left operand and check whether clone changes too
  lhsOperand->castTo<LiteralInt>()->setValue(111);
  ASSERT_EQ(binaryExpression->getLeft()->castTo<LiteralInt>()->getValue(), 111);
  ASSERT_EQ(clonedBinaryExprCasted->getLeft()->castTo<LiteralInt>()->getValue(), 0);

  // make changes to the right operand and check whether clone changes too
  rhsOperand->castTo<LiteralInt>()->setValue(42);
  ASSERT_EQ(binaryExpression->getRight()->castTo<LiteralInt>()->getValue(), 42);
  ASSERT_EQ(clonedBinaryExprCasted->getRight()->castTo<LiteralInt>()->getValue(), 987);

  // make changes to the operator and check whether clone changes too
  *operatore = *new Operator(OpSymb::multiplication);
  ASSERT_TRUE(binaryExpression->getOp()->castTo<Operator>()->equals(OpSymb::multiplication));
  ASSERT_TRUE(clonedBinaryExprCasted->getOp()->equals(OpSymb::addition));
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_LogicalExpr) {  /* NOLINT */
  const bool keepOriginalId = false;

  // create a new logical expression
  auto lhsOperand = new LiteralInt(0);
  auto operatore = new Operator(OpSymb::greater);
  auto rhsOperand = new LiteralInt(987);
  auto logicalExpression = new LogicalExpr(lhsOperand, operatore, rhsOperand);
  // clone the logical expression
  auto clonedLogicalExpression = logicalExpression->clone(keepOriginalId);
  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, logicalExpression, clonedLogicalExpression);

  // make changes to the left operand and check whether clone changes too
  lhsOperand->castTo<LiteralInt>()->setValue(111);
  ASSERT_EQ(logicalExpression->getLeft()->castTo<LiteralInt>()->getValue(), 111);
  auto clonedLogicalExpr = clonedLogicalExpression->castTo<LogicalExpr>();
  ASSERT_EQ(clonedLogicalExpr->getLeft()->castTo<LiteralInt>()->getValue(), 0);

  // make changes to the right operand and check whether clone changes too
  rhsOperand->castTo<LiteralInt>()->setValue(42);
  ASSERT_EQ(logicalExpression->getRight()->castTo<LiteralInt>()->getValue(), 42);
  ASSERT_EQ(clonedLogicalExpr->getRight()->castTo<LiteralInt>()->getValue(), 987);

  // make changes to the operator and check whether clone changes too
  *operatore = *new Operator(OpSymb::smaller);
  ASSERT_TRUE(logicalExpression->getOp()->castTo<Operator>()->equals(OpSymb::smaller));
  ASSERT_TRUE(clonedLogicalExpr->getOp()->equals(OpSymb::greater));
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_If) {  /* NOLINT */
  const bool keepOriginalId = true;

  // create new If object
  auto ifStmtCondition = new LogicalExpr(
      new LiteralInt(12),
      OpSymb::greater,
      new LiteralInt(43));
  auto ifStmtThenBranch = new Block(new VarAssignm("alpha", new LiteralBool(true)));
  auto ifStmtElseBranch = new Block(new VarAssignm("alpha", new LiteralBool(false)));
  auto ifStmt = new If(ifStmtCondition, ifStmtThenBranch, ifStmtElseBranch);

  // check if attributes are set
  ASSERT_EQ(ifStmt->getCondition(), ifStmtCondition);
  ASSERT_EQ(ifStmt->getThenBranch(), ifStmtThenBranch);
  ASSERT_EQ(ifStmt->getElseBranch(), ifStmtElseBranch);

  // clone the object
  If *clonedIfStmt = dynamic_cast<If *>(ifStmt->clone(keepOriginalId));
  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, ifStmt, clonedIfStmt);

  // check if changing the condition in the original also changes the cloned If statement
  ifStmtCondition->setAttributes(new LiteralInt(99),
                                 new Operator(OpSymb::smaller),
                                 new LiteralInt(1));
  // check if changes were applied to the original one
  ASSERT_EQ(ifStmt->getCondition()->castTo<LogicalExpr>()->getLeft()->castTo<LiteralInt>()->getValue(), 99);
  ASSERT_TRUE(ifStmt->getCondition()->castTo<LogicalExpr>()->getOp()->equals(OpSymb::smaller));
  ASSERT_EQ(ifStmt->getCondition()->castTo<LogicalExpr>()->getRight()->castTo<LiteralInt>()->getValue(), 1);
  // check if changes were applied to the cloned one
  auto clonedIfCondition = clonedIfStmt->getCondition()->castTo<LogicalExpr>();
  ASSERT_EQ(clonedIfCondition->getLeft()->castTo<LiteralInt>()->getValue(), 12);
  ASSERT_TRUE(clonedIfCondition->getOp()->equals(OpSymb::greater));
  ASSERT_EQ(clonedIfCondition->getRight()->castTo<LiteralInt>()->getValue(), 43);

  // check if changing the then branch in the original also changes the cloned If statement
  *ifStmtThenBranch = *new Block(new VarAssignm("beta", new LiteralBool(false)));
  auto ifFirstThenStatement = ifStmt->getThenBranch()->castTo<Block>()->getStatements()->front();
  ASSERT_EQ(ifFirstThenStatement->castTo<VarAssignm>()->getIdentifier(), "beta");
  auto clonedIfStmtFirstThenStatement =
      clonedIfStmt->getThenBranch()->castTo<Block>()->getStatements()->front()->castTo<VarAssignm>()->getIdentifier();
  ASSERT_EQ(clonedIfStmtFirstThenStatement, "alpha");

  // check if changing the else branch in the original also changes the cloned If statement
  *ifStmtElseBranch = *new Block(new VarAssignm("gamma", new LiteralBool(true)));
  auto ifFirstStatementElseBranch = ifStmt->getElseBranch()->castTo<Block>()->getStatements()->front();
  ASSERT_EQ(ifFirstStatementElseBranch->castTo<VarAssignm>()->getIdentifier(), "gamma");
  auto clonedIfFirstElseStatement = clonedIfStmt->getElseBranch()->castTo<Block>()->getStatements()->front();
  ASSERT_EQ(clonedIfFirstElseStatement->castTo<VarAssignm>()->getIdentifier(), "alpha");
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_Call) {  /* NOLINT */
  const bool keepOriginalId = false;

  // create new Call object 'call'
  auto callFunctionParam = new FunctionParameter(new Datatype(Types::INT),
                                                 new Variable("pinCode"));
  auto callFunction = new Function("determineSecurityLevel");
  auto call = new Call({callFunctionParam}, callFunction);
  // verify that all parameters are set
  ASSERT_EQ(call->getFunc()->getName(), "determineSecurityLevel");
  ASSERT_EQ(call->getArguments().size(), 1);
  ASSERT_EQ(call->getArguments().front()->getDatatype()->getType(), Types::INT);
  ASSERT_EQ(call->getArguments().front()->getValue()->castTo<Variable>()->getIdentifier(), "pinCode");

  // clone the object
  auto clonedCall = call->clone(keepOriginalId);
  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, call->castTo<AbstractNode>(), clonedCall);

  // test if changing the original FunctionParameter changes the cloned one too
  callFunctionParam->setAttributes(new Datatype(Types::FLOAT), new Variable("var"));
  ASSERT_EQ(clonedCall->castTo<Call>()->getArguments().front()->getDatatype()->getType(), Types::INT);
  ASSERT_EQ(clonedCall->castTo<Call>()->getArguments().front()->getValue()->castTo<Variable>()->getIdentifier(),
            "pinCode");

  // test if changing the original function changes the cloned one too
  ASSERT_EQ(call->getFunc()->getBody().size(), 0);
  ASSERT_EQ(clonedCall->castTo<Call>()->getFunc()->getBody().size(), 0);
  callFunction->addStatement(new VarAssignm("alpha", new LiteralInt(22)));
  ASSERT_EQ(call->getFunc()->getBody().size(), 1);
  ASSERT_EQ(clonedCall->castTo<Call>()->getFunc()->getBody().size(), 0);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_Function) {  /* NOLINT */
  const bool keepOriginalId = true;

  // create new Function object functionStmt
  std::string functionName = "computeSecretKeys";
  std::vector<FunctionParameter *> args;
  auto *funcParam = new FunctionParameter(new Datatype(Types::INT), new Variable("seed"));
  args.push_back(funcParam);
  std::vector<AbstractStatement *> bodyStatements;
  bodyStatements.push_back(new VarAssignm("alpha", new LiteralInt(22)));
  auto functionStmt = new Function(functionName, args, bodyStatements);

  // verify that all parameters are set
  ASSERT_EQ(functionStmt->getName(), functionName);
  ASSERT_EQ(functionStmt->getParams().size(), 1);
  auto functionFirstParam = functionStmt->getParams().front();
  ASSERT_EQ(functionFirstParam, args.front());
  ASSERT_EQ(functionStmt->getBody(), bodyStatements);

  // clone functionStmt as clonedFunctionStmt
  auto clonedFunctionStmt = dynamic_cast<Function *>(functionStmt->clone(keepOriginalId));
  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, functionStmt, clonedFunctionStmt);

  // test if changing the original FunctionParameter changes the cloned one too
  auto clonedFunctionFirstParam = clonedFunctionStmt->getParams().front();
  ASSERT_EQ(clonedFunctionFirstParam->getDatatype()->getType(), Types::INT);
  ASSERT_EQ(clonedFunctionFirstParam->getValue()->castTo<Variable>()->getIdentifier(), "seed");
  funcParam->setAttributes(new Datatype(Types::FLOAT), new Variable("floatThreshold"));
  ASSERT_EQ(functionFirstParam->getDatatype()->getType(), Types::FLOAT);
  ASSERT_EQ(clonedFunctionFirstParam->getDatatype()->getType(), Types::INT);
  ASSERT_EQ(functionFirstParam->getValue()->castTo<Variable>()->getIdentifier(), "floatThreshold");
  ASSERT_EQ(clonedFunctionFirstParam->getValue()->castTo<Variable>()->getIdentifier(), "seed");

  // test if adding a new FunctionParameter to the original Function
  ASSERT_EQ(functionStmt->getParams().size(), 1);
  ASSERT_EQ(clonedFunctionStmt->getParams().size(), 1);
  functionStmt->addParameter(
      new FunctionParameter(new Datatype(Types::INT, true),
                            new Variable("randomNumber")));
  ASSERT_EQ(functionStmt->getParams().size(), 2);
  ASSERT_EQ(clonedFunctionStmt->getParams().size(), 1);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_FunctionParameter) { /* NOLINT */
  const bool keepOriginalId = true;
  auto varExpr = new Variable("alpha");
  auto functionParam = new FunctionParameter(new Datatype(Types::INT), varExpr);
  ASSERT_EQ(functionParam->getDatatype()->getType(), Types::INT);
  ASSERT_EQ(functionParam->getValue(), varExpr);
  auto clonedFunctionParam = dynamic_cast<FunctionParameter *>(functionParam->clone(keepOriginalId));

  // Test if changing original also modifies the copy. As there are no methods to change a FunctionParameter object, we
  // need to use the pointer to change the value pointed to.
  delete functionParam->getDatatype();
  *functionParam->getDatatype() = *new Datatype(Types::FLOAT);
  ASSERT_EQ(functionParam->getDatatype()->getType(), Types::FLOAT);
  ASSERT_EQ(clonedFunctionParam->getDatatype()->getType(), Types::INT);
  delete functionParam->getValue();
  *varExpr = *new Variable("beta");
  ASSERT_EQ(functionParam->getValue()->castTo<Variable>()->getIdentifier(), "beta");
  ASSERT_EQ(clonedFunctionParam->getValue()->castTo<Variable>()->getIdentifier(), "alpha");

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, functionParam, clonedFunctionParam);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_CallExternal) {  /* NOLINT */
  const bool keepOriginalId = false;
  std::vector<FunctionParameter *> functionParams;
  auto *fp = new FunctionParameter(new Datatype(Types::INT), new Variable("blah"));
  functionParams.push_back(fp);

  auto callExternal = new CallExternal("randomFunction", functionParams);
  ASSERT_EQ(callExternal->getFunctionName(), "randomFunction");
  ASSERT_EQ(callExternal->getArguments().size(), 1);
  auto clonedCallExternal =
      dynamic_cast<CallExternal *>(callExternal->clone(keepOriginalId));

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId,
                       static_cast<AbstractExpr *>(callExternal),
                       static_cast<AbstractExpr *>(clonedCallExternal));

  // test if changing the original FunctionParameter vector also modifies the copy
  fp->setAttributes(new Datatype(Types::FLOAT), new Variable("input"));
  ASSERT_EQ(callExternal->getArguments().size(), 1);
  ASSERT_EQ(clonedCallExternal->getArguments().size(), 1);
  ASSERT_EQ(callExternal->getArguments().front()->getDatatype()->getType(), Types::FLOAT);
  auto clonedCallFirstArgument = clonedCallExternal->getArguments().front();
  ASSERT_EQ(clonedCallFirstArgument->getDatatype()->getType(), Types::INT);
  ASSERT_EQ(callExternal->getArguments().front()->getValue()->castTo<Variable>()->getIdentifier(), "input");
  ASSERT_EQ(clonedCallFirstArgument->getValue()->castTo<Variable>()->getIdentifier(), "blah");
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_Block) {  /* NOLINT */
  const bool keepOriginalId = true;
  auto firstStatement = new VarAssignm("alpha", new LiteralInt(222));
  auto blockStatement = new Block(firstStatement);
  ASSERT_EQ(blockStatement->getStatements()->size(), 1);
  ASSERT_EQ(blockStatement->getStatements()->front(), firstStatement);
  auto clonedBlockStatement = dynamic_cast<Block *>(blockStatement->clone(keepOriginalId));

  // test if changing original also modifies the copy
  firstStatement->setAttribute(new LiteralFloat(2221.844f));
  ASSERT_EQ(blockStatement->getStatements()->size(), 1);
  ASSERT_EQ(blockStatement->getStatements()->front(), firstStatement);
  ASSERT_EQ(clonedBlockStatement->getStatements()->size(), 1);
  // check cloned node
  auto clonedVarAssignm = clonedBlockStatement->getStatements()->front()->castTo<VarAssignm>();
  ASSERT_EQ(clonedVarAssignm->getVarTargetIdentifier(), "alpha");
  ASSERT_EQ(clonedVarAssignm->getValue()->castTo<LiteralInt>()->getValue(), 222);

  // delete all statements from original and check whether statements are still in clone
  blockStatement->removeChildren();
  ASSERT_EQ(blockStatement->getStatements()->size(), 0);
  ASSERT_EQ(clonedBlockStatement->getStatements()->size(), 1);

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, blockStatement, clonedBlockStatement);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_LiteralBool) {  /* NOLINT */
  const bool keepOriginalId = true;
  auto lint = new LiteralBool(true);
  ASSERT_EQ(lint->getValue(), true);
  auto clonedNode = dynamic_cast<LiteralBool *>(lint->clone(keepOriginalId));

  // test if changing original also modifies the copy
  lint->setValue(false);
  ASSERT_EQ(lint->getValue(), false);
  ASSERT_EQ(clonedNode->getValue(), true);

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, lint, clonedNode);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_LiteralFloat) {  /* NOLINT */
  const bool keepOriginalId = false;
  auto literalFloat = new LiteralFloat(4.11f);
  ASSERT_EQ(literalFloat->getValue(), 4.11f);
  auto clonedLiteral = dynamic_cast<LiteralFloat *>(literalFloat->clone(keepOriginalId));

  // test if changing original also modifies the copy
  literalFloat->setValue(2.72f);
  ASSERT_EQ(literalFloat->getValue(), 2.72f);
  ASSERT_EQ(clonedLiteral->getValue(), 4.11f);

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, literalFloat, clonedLiteral);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_LiteralInt) {  /* NOLINT */
  const bool keepOriginalId = true;
  auto literalInt = new LiteralInt(4421);
  ASSERT_EQ(literalInt->getValue(), 4421);
  auto clonedLiteral = dynamic_cast<LiteralInt *>(literalInt->clone(keepOriginalId));

  // test if changing original also modifies the copy
  literalInt->setValue(1137);
  ASSERT_EQ(literalInt->getValue(), 1137);
  ASSERT_EQ(clonedLiteral->getValue(), 4421);

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, literalInt, clonedLiteral);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_LiteralString) {  /* NOLINT */
  const bool keepOriginalId = false;
  auto literalStr = new LiteralString("alpha");
  ASSERT_EQ(literalStr->getValue(), "alpha");
  auto clonedLiteral = dynamic_cast<LiteralString *>(literalStr->clone(keepOriginalId));

  // test if changing original also modifies the copy
  literalStr->setValue("gamma");
  ASSERT_EQ(literalStr->getValue(), "gamma");
  ASSERT_EQ(clonedLiteral->getValue(), "alpha");

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, literalStr, clonedLiteral);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_Operator) {  /* NOLINT */
  const bool keepOriginalId = false;
  auto operatore = new Operator(OpSymb::LogCompOp::logicalAnd);
  ASSERT_TRUE(operatore->equals(OpSymb::LogCompOp::logicalAnd));
  auto clonedOp = dynamic_cast<Operator *>(operatore->clone(keepOriginalId));

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, operatore, clonedOp);

  // test if deleting the original leaves the clone untouched
  // (alternative test because we cannot modify the variable identifier)
  delete operatore;
  ASSERT_EQ(clonedOp->getOperatorString(), "AND");
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_Return) {  /* NOLINT */
  const bool keepOriginalId = false;
  auto oldValue = new LiteralInt(944782);
  auto returnStatement = new Return(oldValue);
  ASSERT_EQ(returnStatement->getReturnExpressions().front(), oldValue);
  auto clonedReturn = dynamic_cast<Return *>(returnStatement->clone(keepOriginalId));

  // test if changing original also modifies the copy
  auto newValue = new LiteralFloat(7768.3331f);
  returnStatement->setAttributes({newValue});
  ASSERT_EQ(returnStatement->getReturnExpressions().front()->castTo<LiteralFloat>(), newValue);
  ASSERT_EQ(clonedReturn->getReturnExpressions().front()->castTo<LiteralInt>()->getValue(), oldValue->getValue());

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, returnStatement, clonedReturn);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_UnaryExpr) {  /* NOLINT */
  const bool keepOriginalId = true;
  auto unaryExpr = new UnaryExpr(OpSymb::negation, new LiteralBool(true));
  ASSERT_TRUE(unaryExpr->getOp()->equals(OpSymb::negation));
  auto clonedUnaryExpr = dynamic_cast<UnaryExpr *>(unaryExpr->clone(keepOriginalId));

  // test if changing original also modifies the copy
  unaryExpr->setAttributes(OpSymb::UnaryOp::decrement, new LiteralInt(22));
  ASSERT_TRUE(unaryExpr->getOp()->equals(OpSymb::UnaryOp::decrement));
  ASSERT_EQ(unaryExpr->getRight()->castTo<LiteralInt>()->getValue(), 22);
  // check cloned node
  ASSERT_TRUE(clonedUnaryExpr->getOp()->equals(OpSymb::UnaryOp::negation));
  ASSERT_EQ(clonedUnaryExpr->getRight()->castTo<LiteralBool>()->getValue(), true);

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, unaryExpr, clonedUnaryExpr);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_VarAssignm) {  /* NOLINT */
  const bool keepOriginalId = true;
  auto oldValue = new LiteralFloat(42.11f);
  auto varAssignment = new VarAssignm("alpha", oldValue);
  ASSERT_EQ(varAssignment->getVarTargetIdentifier(), "alpha");
  ASSERT_EQ(varAssignment->getValue()->castTo<LiteralFloat>()->getValue(), oldValue->getValue());
  auto clonedVarAssignm = dynamic_cast<VarAssignm *>(varAssignment->clone(keepOriginalId));

  // test if changing original also modifies the copy
  auto newValue = new LiteralFloat(111.321f);
  varAssignment->setAttribute(newValue);
  ASSERT_EQ(varAssignment->getValue()->castTo<LiteralFloat>()->getValue(), newValue->getValue());
  ASSERT_EQ(clonedVarAssignm->getValue()->castTo<LiteralFloat>()->getValue(), oldValue->getValue());

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, varAssignment, clonedVarAssignm);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_VarDecl) {  /* NOLINT */
  const bool keepOriginalId = true;
  auto varDecl = new VarDecl("alpha", Types::INT, new LiteralInt(2442));
  ASSERT_EQ(varDecl->getVarTargetIdentifier(), "alpha");
  ASSERT_EQ(varDecl->getDatatype()->getType(), Types::INT);
  ASSERT_EQ(varDecl->getInitializer()->castTo<LiteralInt>()->getValue(), 2442);
  auto clonedVarDecl = dynamic_cast<VarDecl *>(varDecl->clone(keepOriginalId));

  // test if changing original also modifies the copy
  varDecl->setAttributes("beta",
                         new Datatype(Types::FLOAT),
                         new LiteralFloat(5.212f));
  ASSERT_EQ(varDecl->getVarTargetIdentifier(), "beta");
  ASSERT_EQ(varDecl->getDatatype()->getType(), Types::FLOAT);
  ASSERT_EQ(varDecl->getInitializer()->castTo<LiteralFloat>()->getValue(), 5.212f);
  ASSERT_EQ(clonedVarDecl->getVarTargetIdentifier(), "alpha");
  ASSERT_EQ(clonedVarDecl->getDatatype()->getType(), Types::INT);
  ASSERT_EQ(clonedVarDecl->getInitializer()->castTo<LiteralInt>()->getValue(), 2442);

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, varDecl, clonedVarDecl);
}

TEST_F(NodeCloneTestFixture, cloneRecursiveDeep_Variable) {  /* NOLINT */
  const bool keepOriginalId = false;
  auto variable = new Variable("secretX");
  ASSERT_EQ(variable->getIdentifier(), "secretX");
  auto clonedVar = dynamic_cast<Variable *>(variable->clone(keepOriginalId));

  // test if all fields belonging to Node class were copied
  assertNodeAttributes(keepOriginalId, variable, clonedVar);

  // test if deleting the original leaves the clone untouched
  // (alternative test because we cannot modify the variable identifier)
  delete variable;
  ASSERT_EQ(clonedVar->getIdentifier(), "secretX");
}
