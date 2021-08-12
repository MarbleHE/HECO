

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
  std::vector<std::string> tokens;
  std::vector<std::string> inputs = {};
  std::vector<std::string> outputs = {};




 public:

  // constructor
  VeriLogToDsl(std::string fileName);


  /// parses entire file into an array of tokens (tokens)
  void tokenizeFile();

  /// parse input from tokenised file
  void parseInput();


  /// parse output
  void parseOutput();


  /// parse program
  void parseAssignment(size_t startIndex, size_t endIndex);

  /// get tokens array (entire file)
  std::vector<std::string> getTokens();

  std::vector<std::string> getInputs();

  std::vector<std::string> getOutputs();


};


#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VERILOGTODSL_H_
