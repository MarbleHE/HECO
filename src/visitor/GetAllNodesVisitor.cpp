#include "ast_opt/visitor/GetAllNodesVisitor.h"

SpecialGetAllNodesVisitor::SpecialGetAllNodesVisitor() = default;;

void SpecialGetAllNodesVisitor::visit(AbstractNode &elem) {
  // std::cout << "Visiting: " << elem.toString(false) << std::endl;
  v.push_back(&elem); //raw pointer
  for (auto &c: elem) { c.accept(*this); }
}

