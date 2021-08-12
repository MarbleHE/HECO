
#include "ast_opt/utilities/VeriLogToDsl.h"

VeriLogToDsl::VeriLogToDsl(std::string fileName) {_filename = fileName;}


void VeriLogToDsl::parseFile() {

  const std::string& whitespace = " \n\r\t\f\v";

  std::ifstream veriLogFile(_filename);

  if (!veriLogFile.is_open()) {
    throw std::runtime_error("File not open.");
  }
  else {// do something with the file, such as parse

    std::cout << "Parsing file: " << _filename << std::endl;

    // tokenise entire file
    std::vector<std::string> tokens;
    for (std::string line; std::getline(veriLogFile, line);) {
      std::istringstream split(line);

      for (std::string each; std::getline(split, each, ' '); tokens.push_back(each));
    }


  // close file
  veriLogFile.close();
  }


}


