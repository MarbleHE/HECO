

#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VERILOGTODSL_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VERILOGTODSL_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
//


class VeriLogToDsl {

 private:
  std::ifstream veriLogFile;
  std::string _filename;
  std::vector<std::string> tokens;
  std::vector<std::string> inputs = {};
  std::vector<std::string> outputs = {};

  ///vector holding assignments:
  /// assignment[0]: result
  /// assignment[1]: operand1
  /// assignment[2]: operator
  /// assignment[3]: operand2
  std::vector<std::vector<std::string>> assignments;

 public:

  // constructor
  VeriLogToDsl(std::string fileName);


  /// parses entire file into an array of tokens (tokens)
  void tokenizeFile();

  /// parse input from tokenised file
  void parseInput();


  /// parse output
  void parseOutput();

  /// find index of next 'assign'
  /// \return index
  int findNextAssignmentBlock(int index);

  /// parse single assignment
  /// \param startIndex index of first element after 'assign'
  /// \param endIndex index of next 'assign'
  /// \return vector containing assignment specified by its indices
  std::vector<std::string> parseSingleAssignment(size_t startIndex, size_t endIndex);

  /// assignment parser. populates the vector of vectors with all assignments.
  void parseAllAssignments();

  /// get tokens array (entire file)
  /// \return tokens
  std::vector<std::string> getTokens();

  /// get inputs vector
  /// \return inputs
  std::vector<std::string> getInputs();

  /// get outputs vector
  /// \return outputs
  std::vector<std::string> getOutputs();

  /// get assignments vector
  /// \return assignments (vector holding vectors with all assignments)
  std::vector<std::vector<std::string>> getAssignments();


};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_VERILOGTODSL_H_
