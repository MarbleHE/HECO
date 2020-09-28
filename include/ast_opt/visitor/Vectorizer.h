#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VECTORIZER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VECTORIZER_H_
#include <set>
#include <unordered_map>
#include "ast_opt/utilities/Datatype.h"
#include "ast_opt/visitor/ScopedVisitor.h"

// Hashing & Equality function specializations to make std::unordered_map work with ScopedIdentifiers as keys
namespace std {
template<>
struct hash<ScopedIdentifier> {
  size_t operator()(const ScopedIdentifier &s) const {
    return std::hash<std::string>{}(s.getScope().getScopeName() + "::" + s.getId());
  }
};

template<>
struct equal_to<ScopedIdentifier> {
  bool operator()(ScopedIdentifier const &s1, ScopedIdentifier const &s2) {
    return s1.getId()==s2.getId() && s1.getScope().getScopeName()==s2.getScope().getScopeName();
  }
};
}

typedef std::unordered_map<ScopedIdentifier, Datatype> TypeMap;

class BatchingConstraint {};
typedef std::unordered_map<ScopedIdentifier, BatchingConstraint> ConstraintMap;

typedef std::unordered_multimap<ScopedIdentifier, int> RotationMap;

class ComplexValue {};
typedef std::set<ComplexValue> ValueSet;

// Forward Declaration for typedef below (must be above documentation, to ensure documentation is associated with the right type)
class SpecialVectorizer;

/// The Vectorizer takes a pre-processed (CTES, etc) AST and greedily searches for opportunities to batch computations.
/// In addition to the AST, it requires Variable information (e.g. scalar? secret? type?)
/// which should be provided as a map between ScopedIdentifiers and Datatype objects
/// Optionally, the vectorizer takes batching constraints as an input. Lacking information, it assumes continuous vectors and free scalars.
/// TODO: Optionally, it should be possible to specify that server-side parameters are "private" and should not be shared to the client in auxiliary information.
///   This would also need an extension of the input language (in addition to new Vectorizer logic) and is out-of-scope for the initial release version.
/// The output is (in addition to the transformed AST), a list of "auxiliary information" about the batching of the input
/// Where, if necessary, new vector-valued input variables are defined from scalar variables that appear in the original program
/// E.g. __input0__ = {a, b, c, d, 0}; **where the last element is the one to be repeated until the ciphertext is full.**
///
/// The Vectorizer does not actually have a notion of ciphertexts, but instead works on vector-valued variables.
/// However, it treats them like ciphertexts in that it does not assume cyclic rotations.
///
/// The Vectorizer relies heavily on batchability-compatibility logic that works roughly as follows:
/// Two AbstractExpressions are batching-compatible if they can be transformed to have the same *Structure*
/// by adding only transparent operations.
/// Two expressions have the same Structure if they have both the same tree structure
/// where we consider literals, array accesses and variables interchangeable for the structure (e.g., (x*3) + 7 <=> (5*z[i]) + a)
/// and all corresponding nodes have compatible batching constraints.
/// Two batching constraints are compatible if at least one is "free" or if both refer to independent slots of the same variable*.
/// For example, assuming standard batching for vectors, x[0] * 5 and x[4] * 7 are compatible.
/// * However, there is one important additional condition: They must be rotation-compatible, too:
/// For example, x[0] * y[1] and x[1] * y[2] would be compatible, but x[0] * y[1] and x[1] * y[3] would NOT
/// More formally, one of the expression's sets of required rotations must be included in the other's set of required rotations.
///
/// The Vectorizer keeps a ConstraintMap, which ensures that batching decisions remain consistent across blocks.
/// It also maintains the TypeMap, accounting for any variables and changes in the program it introduces.
/// In addition, it maintains a map between ScopedIdentifiers and their current "Value"
/// However, in order to support lazy evaluation (see below) and rotation compatibility checks,
/// this is more than just an AbstractExpression. Instead, it could be a set of values + an execution plan (lazy eval)
/// and it could also include information about the already compute rotations for a variable.
///
/// The Vectorizer works on a per-Block level, focussing on assignments,  declarations that include initial values and return statements
/// More complex statements like loops or branching are simply skipped over and the internal blocks considered.
/// Generally, the algorithm works greedily, batching as much as possible as early as possible.
/// However, three components counteract that naive nature:
///
/// Lazy Evaluation:
/// In order to support programs where statements access different indices of the same or different vectors
/// e.g. (x[i] = y[i-1] + z[i+1] + y[i]), we copy the ctxts and rotate them to match.
/// In order to do so optimally, we maintain a set of existing rotations and, for each new required set of indices that need to interact,
/// we calculate the optimal target index that can be achieved by the smallest number of rotations.
/// However, if the statements are of a form that forces everything onto a single target index (e.g. sum = x[i] + y[i-1]),
/// this degrades down to the naive baseline, as each new statement requires a unique rotation offset
/// (For example, if i starts at 0, sum will be in slot 0 so all subsequent statements must be rotated to slot 0).
/// This is bad, because the optimal solution would be log-time rather than linear.
/// Therefore, we introduce lazy evaluation where we postpone operations that would “collapse” the index parallelity.
/// For example, we would represent sum as a set of ctxts + a “plan” (e.g. list of indices to add up) to calculate the final result from them.
/// The plan is only executed when the result needs to interact in an operation that is not batching compatible
/// (e.g. when sum is used in an expression with a different operator/cannot be transformed transparently anymore).
///
/// Rotation-Re-Use:
/// We keep a list of previously calculated rotations, even when they are used as R-Values.
/// We store either the ScopedIdentifier currently containing that rotation, or the rotate() expression where it was created (e.g. if an R-Value or variable overwritten).
/// Should an R-Value-only rotation become useful again, we go back, introduce a new variable initialized to the rotation and replace its use in the original expression.
/// TODO: This requires working parent/etc logic!
///
/// Offset-index-Expression-Re-Use (needs a better name):
/// Sometimes, we can do more than just re-use an already rotated variable. Sometimes, the exact expression we wanted has
/// already been computed by a prior expression, because e.g. we are iterating through some vectors and the structure and offset are constant.
/// Note: In our previous solution (which worked only for loops), we could simply mark an Expression node as "precomputed" and check only if the
/// indices in the current iteration were compatible with the precomputed ones (i.e. that no new rotations were required).
/// However, in our current solution, arbitrary nodes can appear and the fact that no new rotations are required
/// does not imply that two nodes represent the same computation.
/// For example: x[0] = y[15] * 5; and x[2] = y[17] * 7; require the same set of rotations, but neither one is a valid
/// "pre-computation" for the other one.
/// We solve this by comparing expression's structure and content, but considering indices not as their absolute values
/// but as offsets. We could calculate offsets from the target slot if there exists one,
/// or as offsets from the other expression's target slot (if one exists).
/// However, this would make it impossible to use a single pre-computed value to compare expressions.
/// Instead, we define indices as offsets from the first index occurring in-order in the expression AST.
/// Effectively, we  "shift" all indices so that the one occurring first in-order "becomes zero".
/// This way, offset patterns are always the same for compatible expressions, even if the actual indices are different.
///
/// GENERAL APPROACH:
/// Whenever the Vectorizer encounters an assignment statement, it has to answer two questions:
/// First: Is there a specific target slot that the result needs to be written to (e.g. slot i for x[i] = ...)
/// Second: How can the rhs (value) be computed most efficiently (in the target slot, if there is one)?
/// (if the rhs is a binary op and one of the top-level operands is the target, treat op as in-place and look only at other operand)
/// This requires looking at the set of existing expressions and rotations.
/// Potential answers can be "this value is already precomputed", "all rotations exists but a new expression must be calculated",
/// "some new rotations must be created", etc
/// Finally, the Vectorizer emits code (actually AST nodes) that correspond to the required operation and updates its maps & sets.
typedef Visitor<SpecialVectorizer> Vectorizer;

class SpecialVectorizer : public ScopedVisitor {

};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VECTORIZER_H_
