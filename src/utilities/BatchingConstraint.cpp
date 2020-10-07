#include "ast_opt/utilities/BatchingConstraint.h"


BatchingConstraint::BatchingConstraint(int slot, const ScopedIdentifier &identifier)
    : slot(slot), identifier(identifier) {}

int BatchingConstraint::getSlot() const {
  return slot;
}
void BatchingConstraint::setSlot(int slot_) {
  slot = slot_;
}
const ScopedIdentifier &BatchingConstraint::getIdentifier() const {
  return identifier;
}
void BatchingConstraint::setIdentifier(const ScopedIdentifier &identifier_) {
  identifier = identifier_;
}
bool BatchingConstraint::hasTargetSlot() const {
  return getSlot()!=-1;
}
