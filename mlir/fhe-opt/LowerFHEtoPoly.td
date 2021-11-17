#ifndef FHE_LOWER
#define FHE_LOWER

include "FHE/FHEOps.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

def SwitchMultPattern : Pat<(FHE_MultiplyOp $arg1, $arg2, $parms),
                                   (replaceWithValue $arg2)>;


#endif // FHE_LOWER