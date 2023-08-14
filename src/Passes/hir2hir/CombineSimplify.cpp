#include "heco/Passes/hir2hir/CombineSimplify.h"
#include <queue>
#include "heco/IR/FHE/FHEDialect.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void CombineSimplifyPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<fhe::FHEDialect, affine::AffineDialect, scf::SCFDialect, tensor::TensorDialect>();
}

// simplifies
//   %c = fhe.combine(%v#j, %w)
//   %op = fhe.combine(%v#i  %c)
// to
//   %op = fhe.combine(%v#[i,j], %w)
// Technically, the first combine op isn't removed, but if it has no other uses, it'll be canonicalized away, too
void simplify(IRRewriter &rewriter, MLIRContext *context, fhe::CombineOp op)
{
    // Basic sanity check, since we frequently iterate over both things at the same time
    assert(op.getVectors().size() == op.getIndices().size() && "combine op must have indices for each operand");

    // op.getOperation()->getParentOp()->dump();

    if (op.getNumOperands() != 2)
        return;

    if (auto ia = op.getIndices()[0].dyn_cast_or_null<IntegerAttr>())
    {
        if (auto sa = op.getIndices()[1].dyn_cast<StringAttr>())
        {
            if (auto c = op.getVectors()[1].getDefiningOp<fhe::CombineOp>())
            {
                // c->print(llvm::outs());
                // llvm::outs() << "\n";
                // op->print(llvm::outs());
                // llvm::outs() << "\n";
                // llvm::outs().flush();

                if (c.getNumOperands() == 2)
                {
                    if (c.getIndices()[1].dyn_cast<StringAttr>())
                    {
                        // now we have found simple pattern as desired

                        if (auto cia = c.getIndices()[0].dyn_cast_or_null<IntegerAttr>())
                        {
                            // range is just a single index so far
                            if (ia.getInt() == cia.getInt() + 1)
                            {
                                if (op.getVectors()[0] == c.getVectors()[0])
                                {
                                    auto iaa = rewriter.getArrayAttr({ cia, ia });
                                    auto aa = rewriter.getArrayAttr({ iaa, sa });
                                    rewriter.setInsertionPoint(op);
                                    auto new_op = rewriter.replaceOpWithNewOp<fhe::CombineOp>(
                                        op, op->getResultTypes(), c.getVectors(), aa);
                                    // new_op->print(llvm::outs());
                                    // llvm::outs() << "\n";
                                    // llvm::outs().flush();
                                }
                            }
                        }
                        else if (auto iaa = c.getIndices()[0].dyn_cast_or_null<ArrayAttr>())
                        {
                            // there's a range we might need to extend it
                            if (ia.getInt() == iaa[iaa.size() - 1].dyn_cast<IntegerAttr>().getInt() + 1)
                            {
                                if (op.getVectors()[0] == c.getVectors()[0])
                                {
                                    SmallVector<Attribute> sv(iaa.getAsRange<IntegerAttr>());
                                    sv.push_back(ia);
                                    auto iaa = rewriter.getArrayAttr(sv);
                                    auto aa = rewriter.getArrayAttr({ iaa, sa });
                                    rewriter.setInsertionPoint(op);
                                    auto new_op = rewriter.replaceOpWithNewOp<fhe::CombineOp>(
                                        op, op->getResultTypes(), c.getVectors(), aa);
                                    // new_op->print(llvm::outs());
                                    // llvm::outs() << "\n";
                                    // llvm::outs().flush();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//  // Build a list of all inputs %v:i, including (including all %v:i, %v:j coming from a single operand %v:[i,j,..])
//  auto collectInputs =
//      [](fhe::CombineOp op, std::vector<std::pair<Value, IntegerAttr>> &single_inputs, Value &remaining_inputs) {
//        assert(op.getVectors().size()==op.getIndices().size() && "combine op must have indices foreach operand");
//        assert(!remaining_inputs && "when calling collectInputs, remaining_inputs must be set to nullptr!");
//        for (size_t i = 0; i < op.getVectors().size(); ++i) {
//          if (auto aa = op.getIndices()[i].dyn_cast_or_null<ArrayAttr>()) {
//            // if it's a list of indices, they must all be actual indices!
//            for (auto ia: aa.getAsRange<IntegerAttr>()) {
//              assert(ia && "all indices in sublists in fhe.combine must be integers!");
//              single_inputs.emplace_back(op.getVectors()[i], ia);
//            }
//          } else if (auto ia = op.getIndices()[i].dyn_cast_or_null<IntegerAttr>()) {
//            single_inputs.emplace_back(op.getVectors()[i], ia);
//          } else if (auto sa = op.getIndices()[i].dyn_cast_or_null<StringAttr>()) {
//            assert(!remaining_inputs && "There can be only one 'rest' input in an fhe.combine op");
//            remaining_inputs = op.getVectors()[i];
//          }
//        }
//      };

//  std::vector<std::pair<Value, IntegerAttr>> single_inputs;
//  Value remaining_inputs = nullptr; // This is mostly to make sure we only have one "rest" input

//  // Build a list of simplified inputs
//  std::vector<std::pair<Value, IntegerAttr>> new_single_inputs;
//  // For each input, check if it's the result of a combine op (first the single_inputs)
//  for (auto p: single_inputs) {
//    if (auto c = p.first.getDefiningOp<fhe::CombineOp>()) {
//      //  If it is, find out what element of %v = fhe.combine(%u:k, ..., %w) is responsible for slot i
//      updated = true;
//      assert(false && "NOT YET IMPLEMENTED IN COMBINEOP CANONICALIZATION.");

//      //  This is either something like %u:k where k==i, or if i doesn't appear in any of the lists, it's from the
//      remainder: %w:i
//      //  Then replace %v:i with the found %w:i or %u:i (actually we insert into a new list for later use)

//    } else {
//      // Not a combine op, so keep %v:i
//      new_single_inputs.push_back(p);
//    }
//  }
//  // Now we do the same check with the "remaining inputs" operand
//  if (remaining_inputs) {// make sure we're not dereferencing a nullptr, in case everything has an index
//    if (auto c = remaining_inputs.getDefiningOp<fhe::CombineOp>()) {
//      updated = true;
//      // If it's a combine op, we want to collect all the inputs of c and make them our input
//      // However, for each of the inputs of c, we need to check if one of our inputs is overriding that slot

//      // So we begin by collecting a list of all inputs of c:
//      std::vector<std::pair<Value, IntegerAttr>> c_single_inputs;
//      Value c_remaining_inputs = nullptr;
//      collectInputs(c, c_single_inputs, c_remaining_inputs);

//      // Now we can do the overwriting check quite simply:
//      for (auto cp: c_single_inputs) {
//        auto i = cp.second.getInt();
//        bool overwritten = false;
//        for (auto p: single_inputs) {
//          if (i==p.second.getInt()) {
//            overwritten = true;
//            break;
//          }
//        }
//        if (!overwritten) {
//          new_single_inputs.push_back(cp);
//        }
//      }

//      // Finally, once we've promoted all non-overwritten single_inputs,
//      // we replace c as the remaining value with c's remaining value
//      remaining_inputs = c_remaining_inputs;
//    }
//  }

//  // Now go through the list and simplify everything to combine common origins
//  // TODO: The current version only combines sequential bits

//  // First, sort by slot
//  std::sort(new_single_inputs.begin(), new_single_inputs.end(), [](auto &left, auto &right) {
//    return left.second.getInt() < right.second.getInt();
//  });

//  // Now go through and combine ranges
//  llvm::outs() << "///////////////// START RANGE CHECK //////////////////////////\n";
//  llvm::outs().flush();
//  std::vector<std::pair<Value, SmallVector<Attribute>>> aggregated_single_inputs;
//  Value cur_v;
//  for (auto p: new_single_inputs) {
//    if (cur_v==p.first) {
//      updated = true;
//      aggregated_single_inputs.back().second.push_back(p.second);
//      llvm::outs() << "continuation for index " << p.second.getInt() << " of range: ";
//      cur_v.print(llvm::outs());
//      llvm::outs() << "\n";

//    } else {
//      cur_v = p.first;
//      aggregated_single_inputs.push_back({p.first, {p.second}});
//      llvm::outs() << "START for index " << p.second.getInt() << " of range: ";
//      cur_v.print(llvm::outs());
//      llvm::outs() << "\n";
//    }
//  }
//  llvm::outs() << "///////////////// END RANGE CHECK //////////////////////////\n";
//  llvm::outs().flush();

//  // update the op
//  if (updated) {
//    SmallVector<Value> new_vectors;
//    SmallVector<Attribute> new_indices;
//    for (const auto &p: aggregated_single_inputs) {
//      new_vectors.push_back(p.first);
//      new_indices.push_back(ArrayAttr::get(context, p.second));
//    }
//    if (remaining_inputs) {
//      new_vectors.push_back(remaining_inputs);
//      new_indices.push_back(StringAttr::get(context, "all"));
//    }
//    auto new_op = rewriter.create<fhe::CombineOp>(op->getLoc(),
//                                                  op->getResultTypes(),
//                                                  new_vectors,
//                                                  ArrayAttr::get(context, new_indices));
//    //op.erase(); // actually, isn't it the INPUT ops that need erasing?
//    new_op->print(llvm::outs());
//    llvm::outs() << "\n";
//    llvm::outs().flush();
//  } else {
//    //op->print(llvm::outs());
//    //llvm::outs() << "\n";
//    //llvm::outs().flush();
//  }
//}

void CombineSimplifyPass::runOnOperation()
{
    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<fhe::CombineOp>()))
        {
            simplify(rewriter, &getContext(), op);
        }
    }
}