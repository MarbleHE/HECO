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

/// Utility TODO: Find better way to express constant?
def zero : NativeCodeCall<"make_const($_builder,0)">;
def one : NativeCodeCall<"make_const($_builder, 1)">;
def two : NativeCodeCall<"make_const($_builder, 2)">;


//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

/// Transform an FHE mult to a series of polynomial operations
/// x * y turns into
/// t[0] = x[0] * y[0];
/// t[1] = x[0] * y[1] + x[1] * y[0]
/// t[2] = x[1] * y[1]
/// Where we use pmacc for the sum in t[1]
///
/// We use %x0 = poly(x,0) to write x[0] into an SSA value
/// and ctxt(%x0,%x1,%x2) to combine polynomials back into a ctxt.
///
/// The syntax here is functional, so we are essentially writing it like this
/// x*y ==> to_ctxt(
///                 pmul(to_poly(x,0), to_poly(y,0)),
///                 pmacc(to_poly(x,0), to_poly(y,1), pmul(to_poly(x,1), to_poly(y,0)),
///                 pmul(to_poly(x,1), to_poly(y,1))
///         )
///
/// The syntax here is DAG = (OPNAME arg1, arg2, arg3) where the args are again DAGs
def SwitchMultPattern : Pat<(FHE_MultiplyOp $x, $y, $parms),
                      (FHE_MakeCtxtOp (FHE_PolyMulOp $x,$y,$parms), (FHE_PolyAddOp $x,$y,$parms) , (FHE_PolyAddOp $x,$y,$parms))
                    >;


#endif // FHE_LOWER