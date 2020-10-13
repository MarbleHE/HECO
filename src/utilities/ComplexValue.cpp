#include "ast_opt/utilities/ComplexValue.h"

ComplexValue::ComplexValue(AbstractExpression &elem) {
  map.insert({-1, elem});
}

ComplexValue::ComplexValue(AbstractExpression &elem, int slot) {
  map.insert({slot, elem});
}

void ComplexValue::merge(ComplexValue &other) {
  // Get our keys (slots)
  std::vector<int> keys(map.size());
  for (auto const &pair : map) {
    keys.push_back(pair.first);
  }

  // Get other's keys (slots)
  std::vector<int> other_keys(map.size());
  for (auto const &pair : other.map) {
    other_keys.push_back(pair.first);
  }

  // Compute intersection
  std::vector<int> intersection;
  std::set_intersection(keys.begin(),
                        keys.end(),
                        other_keys.begin(),
                        other_keys.end(),
                        std::back_inserter(intersection));

  // Verify that there are no conflicts
  for (auto i : intersection) {
    if (map.at(i).get()!=other.map.at(i).get()) {
      // Requires that they refer to the exact same AST node (object/address comparison)
      throw std::invalid_argument("Cannot merge ComplexValues because they overlap in slot " + std::to_string(i)
                                      + " but have different values.");
    }
  }

  // Perform the actual merge:
  map.merge(other.map);
}