#include "ast_opt/visitor/RuntimeVisitor.h"

#include <utility>
#include "ast_opt/visitor/EvaluationVisitor.h"
#include "ast_opt/visitor/SecretTaintingVisitor.h"
#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VarDecl.h"
#include "ast_opt/ast/MatrixAssignm.h"
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/mockup_classes/Ciphertext.h"

void RuntimeVisitor::visit(Ast &elem) {
  // determine the tainted nodes, i.e., nodes that deal with secret inputs
  SecretTaintingVisitor stv;
  stv.visit(elem);
  auto result = stv.getSecretTaintingList();
  taintedNodesUniqueIds.insert(result.begin(), result.end());

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

  for (elem.getInitializer()->accept(*ev); conditionIsTrue(); elem.getUpdateStatement()->accept(*ev)) {
    elem.getStatementToBeExecuted()->accept(*this);
  }
}

void RuntimeVisitor::visit(MatrixElementRef &elem) {
  Visitor::visit(elem);

  // a helper utility that either returns the value of an already existing LiteralInt (in AST) or performs evaluation
  // and then retrieves the value of the evaluation result (LiteralInt)
  auto determineIndexValue = [&](AbstractExpr *expr) {
    // if index is a literal: simply return its value
    if (auto rowIdxLiteral = dynamic_cast<LiteralInt *>(expr)) return rowIdxLiteral->getValue();
    // if row index is not a literal: evaluate expression
    expr->accept(*ev);
    auto evalResult = ev->getResults().front();
    if (auto evalResultAsLInt = dynamic_cast<LiteralInt *>(evalResult)) {
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
  } else {
    throw std::runtime_error("MatrixElementRef does not refer to a Variable. Cannot continue. Aborting.");
  }

  // store accessed index pair (rowIdx, colidx) and associated variable (matrix) globally
  registerMatrixAccess(varIdentifier, rowIdx, colIdx);
}

RuntimeVisitor::RuntimeVisitor(std::unordered_map<std::string, AbstractLiteral *> funcCallParameterValues) {
  ev = new EvaluationVisitor(std::move(funcCallParameterValues));

}

void RuntimeVisitor::registerMatrixAccess(const std::string &variableIdentifier, int rowIndex, int columnIndex) {
  std::stringstream ss;
  ss << variableIdentifier << "[" << rowIndex << "]"
     << "[" << columnIndex << "]" << "\t(" << currentMatrixAccessMode << ")";
  matrixAccessMap[variableIdentifier][std::pair(rowIndex, columnIndex)] = currentMatrixAccessMode;
//  std::cout << ss.str() << std::endl;
}

void RuntimeVisitor::visit(VarDecl &elem) {
  // Note the counter-intuitive logic here: If getVarValue does *not* throw an exception then the variable has been
  // declared before and we need to inform the user as this will cause issues.
  try {
    static_cast<void>(*ev->getVarValue(elem.getIdentifier()));
    throw std::runtime_error("RuntimeVisitor and EvaluationVisitor cannot handle variable scoping yet. Please use "
                             "unique variable identifiers to allow the RuntimeVisitor to distinguish variables.");
  } catch (std::logic_error &e) {}
  elem.accept(*ev);
}

void RuntimeVisitor::visit(MatrixAssignm &elem) {
  // inlines Visitor::visit(elem);
  Visitor::addStatementToScope(elem);

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

  // collect all (read) matrix accesses on the right-hand side of this assignment and execute the rotations that are
  // required to execute them
  for (auto &[varIdentifier, varAccessMap] : matrixAccessMap) {
    // TODO add check that this variable (varIdentifier) refers to an encrypted variable
    // if (variablesDatatype.at(varIdentifier) != encrypted) continue;

    std::set<int> accessedIndices;
    for (auto &[indexPair, accessMode] : varAccessMap) {
      checkIndexValidity(indexPair);
      accessedIndices.insert(indexPair.second);
    }

    // get rotated ciphertexts that already exist for this variable
    auto existingRotations = varValues.at(varIdentifier);

    // determine the required rotations for this computations
    auto reqRotations = determineRequiredRotations(existingRotations, accessedIndices, targetSlot.second);

    // execute rotations on Ciphertext and add them to the varValues map
    executeRotations(varIdentifier, reqRotations);
  }
  matrixAccessMap.clear();

  // TODO re-visit AST and perform actions using recently generated rotated Ciphertexts and the Ciphertext operations
  std::cout << "stop";
}

void RuntimeVisitor::executeRotations(const std::string &varIdentifier, const std::vector<RotationData> &reqRotations) {
  for (auto rd : reqRotations) {
//    // Check if there is already an existing ciphertext with the same rotation as we need. This should never happen
//    // as the existing rotations should have been passed to determineRequiredRotations before.
//    if (varValues.at(varIdentifier).count(rd.baseCiphertext.getOffsetOfFirstElement()+rd.requiredNumRotations) > 0) {
//      std::cerr << "Requested rotation already exists. This indicates that there's something wrong with your program "
//                   "as you should have passed the existing rotations to determineRequiredRotations." << std::endl;
//      continue; // skip executing this rotation
//    }
    // no need to do anything further as the requested rotation already exists
    if (rd.requiredNumRotations==0) continue;

    // execute rotation on ciphertext
    Ciphertext rotatedCtxt = rd.baseCiphertext.rotate(rd.requiredNumRotations);
    // store rotated ciphertext in varValues
    varValues.at(varIdentifier).emplace(rotatedCtxt.getOffsetOfFirstElement(), rotatedCtxt);
  }
}

std::vector<RotationData> RuntimeVisitor::determineRequiredRotations(std::map<int, Ciphertext> existingRotations,
                                                                     const std::set<int> &reqIndices, int targetSlot) {
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
      resultSet.emplace_back(idx, existingRotations.at(0), targetSlot - idx);
    }
  } else { // existingRotations.size() > 1, i.e., there is more than one existing rotation for this ciphertext
    // if other rotations are existing: figure out the "cheapest" way to align ciphertexts to given target slot based
    // on the existing rotations

    // find the optimal combination of already existing rotated ciphertext + addt. rotations for each required index
    for (auto idx : reqIndices) {
      // a vector that describes how many additional rotations an existing ciphertext would require to align it to
      // the target slot
      std::vector<std::pair<int, Ciphertext>> rotations;

      // determine the difference between current index and existing index for each idx in reqIndices
      bool canReuseExistingRotation = false;
      rotations.reserve(existingRotations.size());
      for (auto[offset, ctxt] : existingRotations) {
        auto numRequiredRotations = targetSlot - (offset + idx);
        rotations.emplace_back(numRequiredRotations, ctxt);
        if (numRequiredRotations==0) {
          // if we can reuse an existing rotation there is no need
          resultSet.emplace_back(idx, ctxt, numRequiredRotations);
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
  elem.getValue()->accept(*this);

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
    Ciphertext ctxt(mx);
    varValues[var->getIdentifier()].emplace(ctxt.getOffsetOfFirstElement(), mx);
  }
}

void RuntimeVisitor::visit(Datatype &elem) {
  Visitor::visit(elem);
}

void RuntimeVisitor::visit(OperatorExpr &elem) {
  Visitor::visit(elem);

  // TODO check if secret variables are involved (integrate the secret tainting visitor!)
  //  if yes: check if we already calculated the required rotations
  //    if yes: execute the operation on the ciphertext
}

