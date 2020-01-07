#ifndef MASTER_THESIS_CODE_INCLUDE_UTILITIES_RANDNUMGEN_H_
#define MASTER_THESIS_CODE_INCLUDE_UTILITIES_RANDNUMGEN_H_

#include <map>
#include <queue>
#include <vector>
#include <string>
#include <random>
#include <utility>
#include "../ast/AbstractStatement.h"
#include "../../include/ast/LiteralInt.h"
#include "../../include/ast/LiteralBool.h"
#include "../../include/ast/LiteralString.h"

class RandLiteralGen {
 private:
  std::mt19937::result_type seed;
  std::mt19937 gen_;
  std::uniform_int_distribution<size_t> distInt_;
  std::uniform_int_distribution<size_t> distBool_;
  std::uniform_int_distribution<size_t> distString_;

  static constexpr const std::pair<int, int> intRange = std::make_pair(0, 999'999);
  static const int stringDefaultMaxLength = 12;
  constexpr static const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  int getRandomIntForStringGen() {
    return distString_(gen_);
  }

  // Credits to Carl from stackoverflow.com (https://stackoverflow.com/a/12468109/3017719)
  std::string random_string(size_t length, std::uniform_int_distribution<size_t> &distString, std::mt19937 &gen) {
    auto randchar = [this]() mutable -> char {
      const size_t max_index = (sizeof(charset) - 1);
      return charset[getRandomIntForStringGen()];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
  }

 public:
  explicit RandLiteralGen(std::mt19937::result_type seed)
      : seed(seed), distInt_{intRange.first, intRange.second}, distBool_{0, 1},
        distString_{0, sizeof(charset) - 2}, gen_(seed) {}

  int getRandomInt() {
    return distInt_(gen_);
  }

  LiteralInt* getRandomLiteralInt() {
    return new LiteralInt(getRandomInt());
  }

  bool getRandomBool() {
    return distBool_(gen_);
  }

  LiteralBool* getRandomLiteralBool() {
    return new LiteralBool(getRandomBool());
  }

  std::string getRandomString(int length) {
    return random_string(length, distString_, gen_);
  }

  LiteralString* getRandomLiteralString(int length = stringDefaultMaxLength) {
    return new LiteralString(getRandomString(length));
  }

  std::map<std::string, Literal*> getRandomValues(const std::map<std::string, Literal*> &vals) {
    std::map<std::string, Literal*> randomizedInputs;
    for (auto &[identifier, lit] : vals) {
      if (dynamic_cast<LiteralBool*>(lit)) {
        randomizedInputs.emplace(identifier, getRandomLiteralBool());
      } else if (dynamic_cast<LiteralString*>(lit)) {
        randomizedInputs.emplace(identifier, getRandomLiteralString());
      } else if (dynamic_cast<LiteralInt*>(lit)) {
        randomizedInputs.emplace(identifier, getRandomLiteralInt());
      }
    }
    return randomizedInputs;
  }
};

#endif //MASTER_THESIS_CODE_INCLUDE_UTILITIES_RANDNUMGEN_H_
