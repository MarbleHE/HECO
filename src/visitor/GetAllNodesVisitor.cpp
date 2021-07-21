#include "ast_opt/visitor/GetAllNodesVisitor.h"

GetAllNodesVisitor::GetAllNodesVisitor() = default;;

void GetAllNodesVisitor::visit(AbstractNode &elem) {
  std::cout << "Visiting: " << elem.toString(false) << std::endl;
  v.push_back(&elem); //raw pointer
  for (auto &c: elem) { c.accept(*this); }
}

