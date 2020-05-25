#include "ast_opt/visitor/RuntimeVisitor.h"
#include <utility>
#include <queue>
#include "ast_opt/visitor/EvaluationVisitor.h"
#include "ast_opt/visitor/SecretTaintingVisitor.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VarDecl.h"
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/AbstractLiteral.h"
#include "ast_opt/ast/MatrixAssignm.h"
#include "ast_opt/ast/VarAssignm.h"
#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/ArithmeticExpr.h"
#include "ast_opt/mockup_classes/Ciphertext.h"
#include <ast_opt/mockup_classes/Plaintext.h>

void RuntimeVisitor::visit(Ast &elem) {
  // determine the tainted nodes, i.e., nodes that deal with secret inputs
  SecretTaintingVisitor stv;
  stv.visit(elem);
  auto result = stv.getSecretTaintingList();
  // create a new set to ensure that remaining values of the last run are overwritten
  taintedNodesUniqueIds = std::unordered_set<std::string>(result.begin(), result.end());

  // start AST traversal
  Visitor::visit(elem);
}

void RuntimeVisitor::visit(For &elem) {
  // inlines Visitor::visit
  auto conditionIsTrue = [&]() -> bool {
    // visit the condition's expression
    elem.getCondition()->accept(*ev);
    // get the expression's evaluation result
    auto cond = *dynamic_cast<LiteralBool *>(ev->getResults().front());
    return cond==LiteralBool(true);
  };

  if (elem.getInitializer()!=nullptr) elem.getInitializer()->accept(*ev);
  for (; conditionIsTrue();) {
    // note: create clones that have the same uniqueNodeId as otherwise the secret tainting IDs won't match anymore
    auto clonedBlock = elem.getBody()->clone(true);
    clonedBlock->accept(*this);
    delete clonedBlock;
    if (elem.getUpdate()!=nullptr) elem.getUpdate()->accept(*ev);

//    std::cout << "== " << ev->getVarValue("x")->castTo<LiteralInt>()->getValue() << ","
//              << ev->getVarValue("y")->castTo<LiteralInt>()->getValue() << std::endl;
  }
}

void RuntimeVisitor::visit(VarAssignm &elem) {
  if (taintedNodesUniqueIds.count(elem.getUniqueNodeId())==0) {
    elem.accept(*ev);
  } else {
    Visitor::visit(elem);
  }
  matrixAccessMap.clear();
}

void RuntimeVisitor::visit(MatrixElementRef &elem) {
  auto intermedResultSize = intermedResult.size();
  Visitor::visit(elem);
  clearIntermediateResults(intermedResultSize);

  // a helper utility that either returns the value of an already existing LiteralInt (in AST) or performs evaluation
  // and then retrieves the value of the evaluation result (LiteralInt)
  auto determineIndexValue = [&](AbstractExpr *expr) {
    // if index is a literal: simply return its value
    if (auto rowIdxLiteral = dynamic_cast<LiteralInt *>(expr)) return rowIdxLiteral->getValue();
    // if row index is not a literal: evaluate expression
    expr->accept(*ev);
    auto evalResult = ev->getResults().front();
    if (auto evalResultAsLInt = dynamic_cast<LiteralInt *>(evalResult)) {
      expr->getOnlyParent()->replaceChild(expr, evalResultAsLInt);
      return evalResultAsLInt->getValue();
    } else {
      throw std::runtime_error("MatrixElementRef row and column indices must evaluate to LiteralInt!");
    }
  };

  // determine value of row and column index
  int rowIdx = determineIndexValue(elem.getRowIndex());
  int colIdx = determineIndexValue(elem.getColumnIndex());

  std::string varIdentifier;
  if (auto var = dynamic_cast<Variable *>(elem.getOperand())) {
    varIdentifier = var->getIdentifier();
  } else if (dynamic_cast<AbstractLiteral *>(elem.getOperand())==nullptr) {
    throw std::runtime_error("MatrixElementRef does not refer to a Variable. Cannot continue. Aborting.");
  }

  if (visitingForEvaluation==EVALUATION_CIPHERTEXT) {
    if (precomputedCiphertexts.count(MatrixElementAccess(rowIdx, colIdx, varIdentifier)) > 0) {
      intermedResult.push(precomputedCiphertexts.at(MatrixElementAccess(rowIdx, colIdx, varIdentifier))->second.ctxt);
    }
  }
  // store accessed index pair (rowIdx, colidx) and associated variable (matrix) globally
  registerMatrixAccess(varIdentifier, rowIdx, colIdx);
}

RuntimeVisitor::RuntimeVisitor(std::unordered_map<std::string, AbstractLiteral *> funcCallParameterValues)
    : visitingForEvaluation(ANALYSIS) {
  ev = new EvaluationVisitor(std::move(funcCallParameterValues));
}

void RuntimeVisitor::registerMatrixAccess(const std::string &variableIdentifier, int rowIndex, int columnIndex) {
//  std::stringstream ss;
//  ss << variableIdentifier << "[" << rowIndex << "]"
//     << "[" << columnIndex << "]" << "\t(" << currentMatrixAccessMode << ")";
  matrixAccessMap[variableIdentifier][std::pair(rowIndex, columnIndex)] = currentMatrixAccessMode;
//  std::cout << ss.str() << std::endl;
}

void RuntimeVisitor::visit(VarDecl &elem) {
  // Note the counter-intuitive logic here: If getVarValue does *not* throw an exception then the variable has been
  // declared before and we need to inform the user as this will cause issues.
//  try {
//    static_cast<void>(*ev->getVarValue(elem.getIdentifier()));
//    throw std::runtime_error("RuntimeVisitor and EvaluationVisitor cannot handle variable scoping yet. Please use "
//                             "unique variable identifiers to allow the RuntimeVisitor to distinguish variables.");
//  } catch (std::logic_error &e) {}

  elem.accept(*ev);

  // if this is a secret variable, create a ciphertext object in varValues
  if (elem.getDatatype()->isEncrypted()) {
    if (elem.getInitializer()!=nullptr) {
      // Handle initialization with given initializer
      Ciphertext *ctxt;
      if (auto valAsLiteralInt = dynamic_cast<LiteralInt *>(elem.getInitializer())) {
        auto vecDoubles = transformToDoubleVector<int>(valAsLiteralInt->getMatrix());
        ctxt = new Ciphertext(vecDoubles);
      } else if (auto valAsLiteralFloat = dynamic_cast<LiteralFloat *>(elem.getInitializer())) {
        auto vecDoubles = transformToDoubleVector<float>(valAsLiteralFloat->getMatrix());
        ctxt = new Ciphertext(vecDoubles);
      } else {
        throw std::logic_error("RuntimeVisitor currently supports only variable declaration of secret int and float "
                               "variables (including matrices).");
      }
      varValues[elem.getIdentifier()].emplace(0, ctxt);
    } else {
      varValues[elem.getIdentifier()].emplace(0, new Ciphertext());
    }
  }
}

void RuntimeVisitor::visit(MatrixAssignm &elem) {
  if (visitingForEvaluation==ANALYSIS) {
    // after the analysis pass the MatrixAssignm's value will already be modified (e.g., MatrixElementRef indices
    // replaced by LiteralInts) thus we only make this copy while visiting this node for the first time
    lastVisitedStatement = elem.clone(true)->castTo<AbstractStatement>();
  }

  // visit the right-hand side of the MatrixAssignm
  currentMatrixAccessMode = WRITE;
  elem.getAssignmTarget()->accept(*this);

  // determine information about the assignment target
  if (matrixAccessMap.size()!=1
      || matrixAccessMap.begin()->second.size()!=1
      || matrixAccessMap.begin()->second.begin()->second!=WRITE) {
    throw std::runtime_error("There should be exactly one matrix write access after visiting the MatrixAssignm's "
                             "left-hand side. Did you forget to clear the matrixAccessMap before visiting the expr?");
  }
  std::string varTargetIdentifier = matrixAccessMap.begin()->first;
  std::pair targetSlot = matrixAccessMap.begin()->second.begin()->first;
  checkIndexValidity(targetSlot);
  matrixAccessMap.clear();

  // visit the right-hand side of the MatrixAssignm
  currentMatrixAccessMode = READ;
  elem.getValue()->accept(*this);

  // check if we already visited this MatrixAssignm before and precomputed the required ciphertexts
  if (visitingForEvaluation==EVALUATION_PLAINTEXT || visitingForEvaluation==EVALUATION_CIPHERTEXT) {
#ifndef NDEBUG
    std::cout << "Skipping recomputation of already existing result..." << std::endl;
#endif
    matrixAccessMap.clear();
    precomputedStatements.erase(elem.getUniqueNodeId());

    // assign the latest result from intermedResult to the respective target slot ciphertext
    varValues[varTargetIdentifier][targetSlot.second]
        = VarValuesEntry(intermedResult.top(), lastVisitedStatement->castTo<MatrixAssignm>()->getValue());

    // empty the intermedResult vector
    // TODO: Do this at the end of all statement visit methods (i.e., AST nodes that inherit from AbstractStatement),
    //  as long as we are using the RuntimeVisitor for the Laplacian sharpening example only, this is fine though.
    clearIntermediateResults();
  } else {
    // if this MatrixAssignm is not tainted then it does not perform any computations on secret variables hence we can
    // simply revisit the operands by after we set te visitingForEvaluation flag
    if (taintedNodesUniqueIds.count(elem.getUniqueNodeId())==0) {
      matrixAccessMap.clear();
      visitingForEvaluation = EVALUATION_PLAINTEXT;
      elem.accept(*this);
      return;
    }

    // a variable to define whether we can skip the additional evaluation pass over this subtree (this is given if we
    // already computed the required expression in a previous iteration)
    bool skipEvaluationPass = false;

    // collect all (read) matrix accesses on the right-hand side of this assignment and execute the rotations that are
    // required to execute them
    for (auto &[varIdentifier, varAccessMap] : matrixAccessMap) {
      // add check that this variable (varIdentifier) refers to an encrypted variable, otherwise continue with next
      // variable
      if (varValues.count(varIdentifier)==0) continue;

      std::unordered_set<int> accessedIndices;
      for (auto &[indexPair, accessMode] : varAccessMap) {
        checkIndexValidity(indexPair);
        accessedIndices.insert(indexPair.second);
      }

      // get rotated ciphertexts that already exist for this variable
      auto existingRotations = varValues.at(varIdentifier);
      // determine the required rotations for this computation
      auto reqRotations = determineRequiredRotations(existingRotations, accessedIndices, targetSlot.second);

      // if all rd of reqRotations have rd.numRotations == 0, we already did computations on this matrix element:
      // if the target ciphertext (e.g., img2[imgSize*x+y]) was computed for the same expression before, i.e., we know
      // that the involved MatrixElementRefs had the same offsets, we also know that we speculatively already computed
      // the expression before -> no need to revisit expression for evaluation pass
      auto allRequiredNumRotationsZero = [&]() -> bool {
        return std::all_of(reqRotations.begin(), reqRotations.end(),
                           [](RotationData &rd) { return rd.requiredNumRotations==0; });
      };
      auto sameExpressionComputed =
          [this, elem](const std::string &varIdentifier, int offset) -> bool {
            if (varValues.count(varIdentifier) > 0) {
              for (auto mapEntry : varValues.at(varIdentifier)) {
                if (mapEntry.second.expr!=nullptr
                    && lastVisitedStatement->castTo<MatrixAssignm>()->getValue()->isEqual(mapEntry.second.expr)) {
                  return true;
                }
              }
            }
            return false;
          };
      auto b = allRequiredNumRotationsZero();
      auto sameExpr = sameExpressionComputed(varTargetIdentifier, targetSlot.second);
      if (b && sameExpr) {
        skipEvaluationPass = true;
      }

      // execute rotations on Ciphertext and add them to the varValues map
      auto rotatedCtxt = executeRotations(varIdentifier, reqRotations);

      // store precomputed ciphertexts by adding a mapping between information in MatrixElementRef (e.g., mx[a][b]) and
      // the associated precoomputed (rotated) ciphertext
      for (auto &varValMapIterator : rotatedCtxt) {
        if (targetSlot.first!=0) throw std::logic_error("Matrix ciphertexts not supported, only row vectors!");
        precomputedCiphertexts.emplace(MatrixElementAccess(targetSlot.first,
                                                           targetSlot.second - varValMapIterator->first,
                                                           varIdentifier), varValMapIterator);
      }

      // mark this MatrixAssignm statement as precomputed, on next visit we can do actual execution on Ciphertexts
      precomputedStatements.insert(elem.getUniqueNodeId());
    }
    // re-visit AST and perform actions using recently generated rotated Ciphertexts and the Ciphertext operations
    matrixAccessMap.clear();
    if (!skipEvaluationPass) {
      visitingForEvaluation = EVALUATION_CIPHERTEXT;
      elem.accept(*this);
      visitingForEvaluation = ANALYSIS;
    } else {
#ifndef NDEBUG
      std::cout << "Skipping evaluation pass..." << std::endl;
#endif
    }
  }
}

std::vector<VarValuesMap::iterator>
RuntimeVisitor::executeRotations(const std::string &varIdentifier, const std::vector<RotationData> &reqRotations) {
  std::vector<VarValuesMap::iterator> ciphertexts;
  for (auto rd : reqRotations) {
    // no need to do anything further as the requested rotation already exists
    if (rd.requiredNumRotations==0) {
      ciphertexts.emplace_back(varValues.at(varIdentifier).find(rd.baseCiphertext->getOffsetOfFirstElement()));
      continue;
    }

#ifndef NDEBUG
    std::cout << "Executing rotations..." << std::endl;
#endif

    // execute rotation on ciphertext
    auto rotatedCtxt = new Ciphertext(rd.baseCiphertext->rotate(rd.requiredNumRotations));
    // store rotated ciphertext in varValues
    auto[it, inserted] = varValues.at(varIdentifier).emplace(rotatedCtxt->getOffsetOfFirstElement(), rotatedCtxt);
    ciphertexts.emplace_back(it);
  }
  return ciphertexts;
}

std::vector<RotationData> RuntimeVisitor::determineRequiredRotations(VarValuesMap existingRotations,
                                                                     const std::unordered_set<int> &reqIndices,
                                                                     int targetSlot) {
  if (targetSlot==-1) {
    // TODO: Implement logic to find the best suitable targetSlot to realize the given reqIndices.
    throw std::logic_error("Logic to automatically determine the best targetSlot is not implemented yet! Please "
                           "specify a targetSlot and re-run the RuntimeVisitor.");
  }

  // Note that the resulting std::set<RotationData> always contains references to the original Ciphertext objects,
  // passed as parameter existingRotations. This is fine as our Ciphertext's rotation method does not rotate in-place
  // but returns a rotated copy instead. This explains why there is no need to clone the ciphertext at this point.

  // if only a rot-0 ctxt is given: create all rotations for reqIndices
  std::vector<RotationData> resultSet;
  if (existingRotations.size()==1) {
    // make sure that the existing rotation is the 0-ciphertext, i.e., the initial (unrotated) ciphertext
    if (existingRotations.count(0)!=1) {
      throw std::runtime_error("There is only one ciphertext in the given map of existing rotations but that is "
                               "not the initial ciphertext with offset zero. Cannot continue.. aborting.");
    }
    // create rotations for each index in reqIndices
    for (auto idx : reqIndices) {
      resultSet.emplace_back(idx, existingRotations.at(0).ctxt, targetSlot - idx);
    }
  } else { // existingRotations.size() > 1, i.e., there is more than one existing rotation for this ciphertext
    // if other rotations are existing: figure out the "cheapest" way to align ciphertexts to given target slot based
    // on the existing rotations

    // find the optimal combination of already existing rotated ciphertext + addt. rotations for each required index
    for (auto idx : reqIndices) {
      // a vector that describes how many additional rotations an existing ciphertext would require to align it to
      // the target slot
      std::vector<std::pair<int, Ciphertext *>> rotations;

      // determine the difference between current index and existing index for each idx in reqIndices
      bool canReuseExistingRotation = false;
      rotations.reserve(existingRotations.size());
      for (auto[offset, varValEntry] : existingRotations) {
        auto numRequiredRotations = targetSlot - (offset + idx);
        rotations.emplace_back(numRequiredRotations, varValEntry.ctxt);
        if (numRequiredRotations==0) {
          // if we can reuse an existing rotation there is no need
          resultSet.emplace_back(idx, varValEntry.ctxt, numRequiredRotations);
          canReuseExistingRotation = true;
          break;
        }
      }

      // TODO: Find the combination of (existing ciphertext, needed rotations) that is "cheapest".
      //  Implement technique used in SEAL that uses bit representation and allows determining cheapest rotation.
      //  Take the values in the set rotations and push the determine cheapest one into resultSet.
      if (!canReuseExistingRotation) {
        throw std::runtime_error(
            "Not implemented: Cannot determine the optimal (ciphertext, req. rotations) set yet.");
      }
    }
  }
  return resultSet;
}

void RuntimeVisitor::checkIndexValidity(std::pair<int, int> rowColumnIndex) {
  if (rowColumnIndex.first!=0) {
    throw std::runtime_error("Currently, the RuntimeVisitor only supports matrix accesses of row vectors (i.e., "
                             "[0][x]) for secret variables. Please rewrite your program to use a row-concatenated "
                             "vector of dimension (1,x) instead.");
  }
}

void RuntimeVisitor::visit(FunctionParameter &elem) {
  // inline Visitor::visit(elem);
  elem.getDatatype()->accept(*this);
//  elem.getValue()->accept(*this);

  // create new Ciphertext if this FunctionParameter refers to an encrypted variable (secret input)
  if (elem.getDatatype()->isEncrypted()) {
    auto var = dynamic_cast<Variable *>(elem.getValue());
    if (var==nullptr) {
      throw std::runtime_error("FunctionParameter's value is expected to be of type Variable!");
    }

    // transform the function parameter's value (need to be passed to RuntimeVisitor's constructor) from Matrix<int>
    // or Matrix<float> to Matrix<double> as ciphertexts currently only support Matrix<double>
    std::vector<double> mx;
    auto value = ev->getVarValue(var->getIdentifier());
    if (auto lint = dynamic_cast<LiteralInt *>(value)) {
      mx = transformToDoubleVector<int>(lint->getMatrix());
    } else if (auto lfloat = dynamic_cast<LiteralFloat *>(value)) {
      mx = transformToDoubleVector<float>(lfloat->getMatrix());
    } else {
      throw std::runtime_error("Only LiteralInt and LiteralFloat ciphertexts supported yet!");
    }

    // create and store ciphertextt in varValues map
    auto ctxt = new Ciphertext(mx);
    varValues[var->getIdentifier()][ctxt->getOffsetOfFirstElement()].ctxt = ctxt;
  }
}

void RuntimeVisitor::visit(Datatype &elem) {
  Visitor::visit(elem);
}

void RuntimeVisitor::visit(ArithmeticExpr &elem) {
  Visitor::visit(elem);

  if (taintedNodesUniqueIds.count(elem.getUniqueNodeId()) > 0) {

  } else {
    ev->visit(elem);
  }
}

void RuntimeVisitor::visit(OperatorExpr &elem) {
  int intermedResultsSize = intermedResult.size();
  if (visitingForEvaluation==EVALUATION_CIPHERTEXT) {
    Visitor::visit(elem);

    // Results from subexpressions (i.e., operands of this OperatorExpr) are pushed to a stack, hence the last
    // evaluated expression in on the top of the stack. As we need to add the elements in reverse order to the
    // operands vector, we use a queue that allows us to deque elements in the order the elements were added.
    std::deque<Ciphertext *> intermedResultReversed;
    for (size_t i = intermedResult.size() - intermedResultsSize; i > 0; --i) {
      intermedResultReversed.push_back(intermedResult.top());
      intermedResult.pop();
    }

    std::vector<Ciphertext *> operands;
    // create a Plaintext for any involved plaintext literal to allow operation exec between Ciphertext and Plaintext
    for (auto opnd : elem.getOperands()) {
      if (auto literalInt = dynamic_cast<LiteralInt *>(opnd)) {
        operands.push_back(new Plaintext(literalInt->getValue()));
      } else if (auto literalFloat = dynamic_cast<LiteralFloat *>(opnd)) {
        operands.push_back(new Plaintext(literalFloat->getValue()));
      } else if (dynamic_cast<OperatorExpr *>(opnd) || dynamic_cast<MatrixElementRef *>(opnd)) {
        // in this case the OperatorExpr/MatrixElementRef was visited before and its Ciphertext was pushed to
        // intermedResult
        operands.push_back(intermedResultReversed.back());
        intermedResultReversed.pop_back();
      } else {
        throw std::runtime_error("Encountered an operand in OperatorExpr that cannot be handled yet!");
      }
    }

    // make sure that we collected all operands
    assert(operands.size()==elem.getOperands().size());

#ifndef NDEBUG
    std::cout << "Computing expression on ciphertext..." << std::endl;
#endif

    // execute operation
    auto it = operands.begin();
    auto result = new Ciphertext(**it);
    for (++it; it!=operands.end(); ++it) {
      if (elem.getOperator()->equals(ADDITION)) {
        *result = (*result) + *(*it);
      } else if (elem.getOperator()->equals(SUBTRACTION)) {
        *result = (*result) - *(*it);
      } else if (elem.getOperator()->equals(MULTIPLICATION)) {
        *result = (*result)**(*it);
      } else if (elem.getOperator()->equals(DIVISION)) {
        *result = (*result)/(*(*it));
      } else {
        throw std::logic_error("Operator not implemented yet!");
      }
    }

    // push the result up to the parent - decision of what to do is to be taken by the statement
    clearIntermediateResults(intermedResultsSize);
    intermedResult.push(result);
  } else if (visitingForEvaluation==EVALUATION_PLAINTEXT) {  //  visitingForEvaluation = EVALUATION_PLAINTEXT
    throw std::logic_error("OperatorExpr evaluation of plaintext (Literals) not implemented yet!");
    // visit the expression using the EvaluationVisitor
    // replace this node in its parent by the evaluation result
  } else {
    Visitor::visit(elem);
  }
}

void RuntimeVisitor::clearIntermediateResults(int numElementsBeforeVisitingChildren) {
  int removeNumElements = (numElementsBeforeVisitingChildren==-1) ?
                          intermedResult.size() :
                          intermedResult.size() - numElementsBeforeVisitingChildren;
  while (removeNumElements > 0) {
    intermedResult.pop();
    --removeNumElements;
  }
}

void RuntimeVisitor::visit(Return &elem) {
  int intermedResultSize = intermedResult.size();
  Visitor::visit(elem);
  while (intermedResult.size() - intermedResultSize > 0) {
    returnValues.push_back(intermedResult.top());
    intermedResult.pop();
  }
}

void RuntimeVisitor::visit(Variable &elem) {
  Visitor::visit(elem);

  // TODO: Find a better way to manage how we distinguish subexpression ciphertexts from those that actually
  //  represent a variable's most recent value. For now, we just return the first Ciphertext that has an associated
  //  expression (this works fine for img2 but would break for img).
  if (taintedNodesUniqueIds.count(elem.getUniqueNodeId()) > 0 && varValues.count(elem.getIdentifier()) > 0) {
    for (auto varValMap : varValues.at(elem.getIdentifier())) {
      if (varValMap.second.expr!=nullptr) {
        intermedResult.push(varValMap.second.ctxt);
        return;
      }
    }
  } else {
    // TODO: Retrieve value from EvaluationVisitor's map (getVarValue).
    elem.getOnlyParent()->replaceChild(&elem, ev->getVarValue(elem.getIdentifier())->clone(true));
  }
}

const std::vector<Ciphertext *> &RuntimeVisitor::getReturnValues() const {
  return returnValues;
}
