

#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VERILOGTODSL_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VERILOGTODSL_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>


class VeriLogToDsl {

 private:
  std::ifstream veriLogFile;
  std::string _filename;
  std::vector<std::string> inputs = {};
  std::vector<std::string> outputs = {};



 public:

  // constructor
  VeriLogToDsl(std::string fileName);


  /// parse file
  void parseFile();

  /// parse input
  void parseInput();


  /// parse output
  void parseOutput();


  /// parse program
  void parseProgram();



};


#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VERILOGTODSL_H_
