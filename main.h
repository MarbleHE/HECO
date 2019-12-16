
#ifndef MASTER_THESIS_CODE_MAIN_H
#define MASTER_THESIS_CODE_MAIN_H

#define _DEBUG_RUNNING() std::cerr << "[" << __FILE_NAME__ << ":" << __LINE__ << "]"  << " ▶︎ Running "<< __PRETTY_FUNCTION__ << "..." << std::endl << std::flush;

#include <iostream>
#include "include/ast/LiteralInt.h"
#include "include/ast/LiteralString.h"
#include "include/ast/VarDecl.h"
#include "include/ast/Variable.h"
#include "include/ast/Block.h"
#include "include/ast/BinaryExpr.h"
#include "include/ast/VarAssignm.h"
#include "include/ast/Group.h"
#include "include/ast/If.h"
#include "include/ast/Return.h"
#include "include/ast/Function.h"
#include "include/ast/FunctionParameter.h"
#include "include/ast/AbstractStatement.h"
#include "include/ast/While.h"
#include "include/ast/LogicalExpr.h"
#include "include/ast/LiteralBool.h"
#include "include/ast/UnaryExpr.h"
#include "include/ast/CallExternal.h"


/// Program's entry point.
int main();

#endif //MASTER_THESIS_CODE_MAIN_H
