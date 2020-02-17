#include "MultDepthVisitor.h"
#include "LogicalExpr.h"
#include "BinaryExpr.h"
#include "VarAssignm.h"
#include "VarDecl.h"
#include "Return.h"

void MultDepthVisitor::visit(BinaryExpr &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(Block &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(Call &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(CallExternal &elem) {
  throw std::logic_error("Depth calculations for ASTs CallExternal nodes not supported yet!");
}

void MultDepthVisitor::visit(Function &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(FunctionParameter &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(If &elem) {
  throw std::logic_error("Depth calculations for ASTs with non-circuit nodes (e.g., If) not supported yet!");
  //Visitor::visit(elem);
}

void MultDepthVisitor::visit(LiteralBool &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(LiteralInt &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(LiteralString &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(LiteralFloat &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(LogicalExpr &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(Operator &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(Return &elem) {
  for (uint i = 0; i < elem.getReturnExpressions().size(); i++) {
    analyzeMultiplicativeDepth("<ReturnValue" + std::to_string(i) + ">",
                               &elem,
                               elem.getReturnExpressions().at(i));
  }
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(VarAssignm &elem) {
  analyzeMultiplicativeDepth(elem.getIdentifier(), &elem, elem.getValue());
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(VarDecl &elem) {
  analyzeMultiplicativeDepth(elem.getIdentifier(), &elem, elem.getInitializer());
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(Variable &elem) {
  Visitor::visit(elem);
}

void MultDepthVisitor::visit(While &elem) {
  throw std::logic_error("Depth calculations for ASTs with non-circuit nodes (e.g., While) not supported yet!");
}

void MultDepthVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
  if (verbose) std::cout << "Call now function getMaxDepth() to retrieve the calculated depth" << std::endl;
}

void MultDepthVisitor::analyzeMultiplicativeDepth(const std::string &varIdentifier,
                                                  AbstractStatement *stmt,
                                                  AbstractExpr *initializer) {
  // if VarDecl contains BinaryExpr or contains LogicalExpr
  std::string initializerNodeType = initializer->getNodeName();
  if (initializerNodeType==BinaryExpr().getNodeName() || initializerNodeType==LogicalExpr().getNodeName()) {
    // count the number of logicalAnd and multiplication operations in the initializer (=: numMults)
    int numMults = initializer->countByTemplate(new BinaryExpr(OpSymb::multiplication))
        + initializer->countByTemplate(new LogicalExpr(OpSymb::logicalAnd));
    // determine the maximum depth of the variables contained in the initializer (=: maxDepth)
    const std::vector<std::string> &variablesList = initializer->getVariableIdentifiers();
    if (variablesList.empty()) {
      // update depth counter by numMults
      updateDepthStructures(stmt, varIdentifier, numMults);
      //depthsPerVariable[varIdentifier] = numMults;
    } else {
      auto maxDepthElement =
          std::max_element(variablesList.begin(), variablesList.end(),
                           [this](std::string const &lhs, std::string const &rhs) {
                             return getDepth(lhs) < getDepth(rhs);
                           });
      // update depth counter by maxDepth + numMults
      //depthsPerVariable[varIdentifier] = getDepth(*maxDepthElement) + numMults;
      updateDepthStructures(stmt, varIdentifier, getDepth(*maxDepthElement) + numMults);
    }
  }
}

/// Extract the max depth of the map of depths.
/// \return The maximum depth of the visited AST.
int MultDepthVisitor::getMaxDepth() {
  // if there is no variable we implicitly have depth 0
  if (depthsPerVariable.empty()) return 0;

  // Credits to Rob from stackoverflow.com (https://stackoverflow.com/a/9371137/3017719)
  using pair_type = decltype(depthsPerVariable)::value_type;
  auto pr = std::max_element(
      std::begin(depthsPerVariable),
      std::end(depthsPerVariable),
      [](const pair_type &p1, const pair_type &p2) {
        return p1.second < p2.second;
      });
  return pr->second;
}

int MultDepthVisitor::getDepth(const std::string &nodeName) {
  auto result = depthsPerVariable.find(nodeName);
  return (result!=depthsPerVariable.end()) ? result->second : 0;
}

MultDepthVisitor::MultDepthVisitor(bool b) : verbose(b) {}

void MultDepthVisitor::updateDepthStructures(AbstractStatement *stmt, const std::string &varIdentifier, int depth) {
  // update (overwrite, if existing) max depth of variable
  depthsPerVariable[varIdentifier] = depth;
  // store depth of this statement
  depthsPerStatement[stmt] = std::make_pair(varIdentifier, depth);
}
