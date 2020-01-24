#include <Operator.h>
#include <queue>
#include "../optimizer/ConeRewriter.h"
#include "genAstDemo.h"
#include "Function.h"
#include "Call.h"
#include "LogicalExpr.h"
#include "Return.h"

using namespace std;

// C++ template to print vector container elements
ostream &operator<<(ostream &os, const vector<Node*> &v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i]->getUniqueNodeId();
    if (i != v.size() - 1)
      os << ", ";
  }
  os << "]";
  return os;
}

void printAstAsTree(const Ast &ast) {
  deque<pair<Node*, int>> q;
  q.emplace_back(ast.getRootNode(), 1);
  while (!q.empty()) {
    auto curNode = q.front().first;
    auto il = q.front().second;
    q.pop_front();
    string outStr;

    cout << string(il*3, ' ') << curNode->getUniqueNodeId()  << endl;
    cout << string(il*3, ' ') << " ↳children: " << curNode->getChildren() << endl;
    cout << string(il*3, ' ') << " ↳parents: " << curNode->getParents() << endl << endl ;
    for_each(curNode->getChildren().rbegin(), curNode->getChildren().rend(), [&q, il](Node* n) {
      q.emplace_front(n, il+1);
    });
  }
}

int main() {
  //runInteractiveDemo();

  Ast ast;
  Function* f = new Function("demoCkt");
  f->addStatement(new Return(new LogicalExpr(
      new LogicalExpr(
          new LogicalExpr(
              new LogicalExpr(
                  new Variable("a_1^(1)"),
                  OpSymb::logicalAnd,
                  new Variable("a_2^(1)")),
              OpSymb::logicalXor,
              new LogicalExpr(
                  new Variable("a_1^(2)"),
                  OpSymb::logicalAnd,
                  new Variable("a_2^(2)"))),
          OpSymb::logicalXor,
          new Variable("y_1")),
      OpSymb::logicalAnd,
      new Variable("a_t"))));

  ast.setRootNode(f);

  printAstAsTree(ast);

  std::cout << std::endl;
  std::cout << "######################" << std::endl;
  std::cout << std::endl;
  std::cout << "isValidCircuit: " << ConeRewriter::isValidCircuit(ast) << std::endl;

  ConeRewriter::reverseEdges(ast);
  std::cout << std::endl;
  std::cout << "getMultDepth: " << ConeRewriter::getMultDepth(ast.getRootNode()) << std::endl;
  std::cout << std::endl;
  std::cout << *ConeRewriter::getPred(ast.getRootNode()) << std::endl;
  std::cout << std::endl;
  //std::cout << *ConeRewriter::getPred(ast.getReducibleCones(ast.getRootNode())) << std::endl;


  return 0;
}

