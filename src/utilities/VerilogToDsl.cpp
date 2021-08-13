
#include "ast_opt/utilities/VeriLogToDsl.h"

VeriLogToDsl::VeriLogToDsl(std::string fileName) {_filename = fileName;}

void VeriLogToDsl::tokenizeFile() {


  std::ifstream veriLogFile(_filename);

  if (!veriLogFile.is_open()) {
    throw std::runtime_error("File not open.");
  }
  else {
    std::cout << "Parsing file: " << _filename << std::endl;

    // tokenise entire file
    for (std::string line; std::getline(veriLogFile, line);) {
      std::istringstream split(line);
      for (std::string each; std::getline(split, each, ' '); tokens.push_back(each));
    }
  // close file
  veriLogFile.close();
  }
}

void VeriLogToDsl::parseInput() {

   if (tokens.empty()) {
     throw std::runtime_error("Array of tokens is empty. Unable to parse inputs.");
   }

   // find start and end indices of input block
   size_t input_start;
   size_t input_end;
   for (int i = 0; i < tokens.size(); i++) {
     if (tokens[i] == "input") { input_start = i;}
     if (tokens[i] == "output") { input_end = i;}
   }

   // put (formatted) inputs into inputs vector
   for (int j = input_start + 1; j < input_end; j++ ) {
     if (tokens[j] != "," && tokens[j] != ";" && tokens[j].size() > 0) {
       std::string formatted_token = tokens[j];
       formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), '\\'), formatted_token.end());
       formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), '['), formatted_token.end());
       formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), ']'), formatted_token.end());
       formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), ';'), formatted_token.end());
       formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), ','), formatted_token.end());
       inputs.push_back(formatted_token);
     }
   }
}

void VeriLogToDsl::parseOutput(){

  if (tokens.empty()) {
    throw std::runtime_error("Array of tokens is empty. Unable to parse outputs.");
  }
  // find start and end indices of output block
  size_t output_start;
  size_t output_end;
  for (int i = 0; i < tokens.size(); i++) {
    if (tokens[i] == "output") { output_start = i;}
    if (tokens[i] == "wire") { output_end = i;}
  }

  // put (formatted) inputs into inputs vector
  for (int j = output_start + 1; j < output_end; j++ ) {
    if (tokens[j] != "," && tokens[j] != ";" && tokens[j].size() > 0) {
      std::string formatted_token = tokens[j];
      formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), '\\'), formatted_token.end());
      formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), '['), formatted_token.end());
      formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), ']'), formatted_token.end());
      formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), ';'), formatted_token.end());
      formatted_token.erase(remove(formatted_token.begin(), formatted_token.end(), ','), formatted_token.end());
      outputs.push_back(formatted_token);
    }
  }
}

int VeriLogToDsl::findNextAssignmentBlock(int index) {
  if (tokens.empty()) {
    throw std::runtime_error("Array of tokens is empty. Unable to parse assignments.");
  }
  // find start and end indices of output block
  size_t first_assign_index;

  int i = index + 1;
  // search for next 'assign'or EOF
  while (tokens[i] != "assign" && i < tokens.size()) {
    i++;
  }
  return i;
}

std::vector<std::string> VeriLogToDsl::parseSingleAssignment(size_t startIndex,size_t endIndex) {

  std::vector<std::string> assignment;

  for(int  i = startIndex; i < endIndex; i++) {
    assignment.push_back(tokens[i]);
  }

  // clean up
  for (int j = 0; j < assignment.size(); j++) {
    assignment[j].erase(remove( assignment[j].begin(),  assignment[j].end(), '\\'),  assignment[j].end());
    assignment[j].erase(remove( assignment[j].begin(),  assignment[j].end(), '['),  assignment[j].end());
    assignment[j].erase(remove( assignment[j].begin(),  assignment[j].end(), ']'),  assignment[j].end());
    assignment[j].erase(remove( assignment[j].begin(),  assignment[j].end(), ';'),  assignment[j].end());
    assignment[j].erase(remove( assignment[j].begin(),  assignment[j].end(), ','),  assignment[j].end());
    }
  // erase whitespace, '=' elements
  std::vector<std::string>::iterator i = assignment.begin();
  while(i != assignment.end()) {
    if(i->find('=', 0) != std::string::npos)
    {
      i = assignment.erase(i);
    }
    else
    {
      ++i;
    }
  }
  std::vector<std::string>::iterator j = assignment.begin();
  while(j != assignment.end()) {
    if(j->find(" \t\r\n", 0) != std::string::npos)
    {
      j = assignment.erase(j);
    }
    else
    {
      ++j;
    }
  }
  // erase empty elements (need to run this twice for some reason)
  for (int i=0; i < assignment.size(); i++)
  {
    if ( assignment[i].empty() ) assignment.erase( assignment.begin() + i );
  }
  for (int i=0; i < assignment.size(); i++)
  {
    if ( assignment[i].empty() ) assignment.erase( assignment.begin() + i );
  }

  return assignment;
}



std::vector<std::string> VeriLogToDsl::getTokens() {
  return tokens;
}

std::vector<std::string> VeriLogToDsl::getInputs() {
  return inputs;
}

std::vector<std::string> VeriLogToDsl::getOutputs() {
  return outputs;
}



