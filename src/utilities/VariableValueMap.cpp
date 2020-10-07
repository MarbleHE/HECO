#include "ast_opt/utilities/VariableValueMap.h"

VariableValueMap::VariableValueMap(const std::unordered_map<ScopedIdentifier, ComplexValue &> &map) : map(map) {}

void VariableValueMap::add(ScopedIdentifier s, ComplexValue &cv) {
  map.insert({s, cv});
  changed.insert(s);
}

const ComplexValue &VariableValueMap::get(const ScopedIdentifier &s) const {
  return map.find(s)->second;
}

ComplexValue &VariableValueMap::getToModify(const ScopedIdentifier &s) {
  changed.insert(s);
  return map.find(s)->second;
}

ComplexValue &VariableValueMap::take(const ScopedIdentifier &s) {
  auto it = map.find(s);
  ComplexValue &cv = it->second;
  map.erase(it);
  auto changed_it = changed.find(s);
  if (changed_it!=changed.end()) {
    changed.erase(changed_it);
  }
  return cv;
}
void VariableValueMap::update(const ScopedIdentifier &s, ComplexValue &cv) {
  map.find(s)->second = cv;
  changed.insert(s);
}
bool VariableValueMap::has(const ScopedIdentifier &s) {
  return map.find(s)!=map.end();
}
void VariableValueMap::resetChangeFlags() {
  changed.clear();
}
std::unordered_set<ScopedIdentifier> VariableValueMap::changedEntries() const {
  return changed;
}