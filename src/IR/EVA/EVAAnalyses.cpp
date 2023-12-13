#include "heco/IR/EVA/EVADialect.h"

using namespace mlir;
using namespace heco;
using namespace eva;

//===----------------------------------------------------------------------===//
// EVA Scale Analysis
//===----------------------------------------------------------------------===//

 int ScaleAnalysis::argument_scale;
 int ScaleAnalysis::waterline;
 int ScaleAnalysis::scale_drop;

ScaleAnalysis::ScaleAnalysis(Operation *op, AnalysisManager &am)
{
    std::vector<Operation *> worklist;
    for (auto &region : op->getRegions())
    {
        region.walk([&worklist](Operation *op) {
            /* process only EVA Dialect operations, to discard the top level operation */
            auto dial = op->getDialect();
            if (dial and dial->getNamespace() == EVADialect::getDialectNamespace())
            {
                worklist.push_back(op);
            }
        });
    }

    for (auto op : worklist)
    {
        int implied_result_scale = calc_scale(op, true);
        
        rescales_needed[op] = 0;
        std::string op_name = op->getName().getStringRef().str();
        // only multiplication can increase the scale
        if (op_name == "eva.multiply") {
            while (implied_result_scale > waterline) {
                rescales_needed[op] += 1;
                implied_result_scale -= scale_drop;
            }
        }

        implied_scales[op] = implied_result_scale;

        int no_implied_result_scale = calc_scale(op, false);

        if (op_name == "eva.rescale") {
            no_implied_result_scale -= scale_drop;
        }

        real_scales[op] = no_implied_result_scale;
    }
}

bool ScaleAnalysis::isInvalidated(const AnalysisManager::PreservedAnalyses &pa)
{
    return true;
}

int ScaleAnalysis::getScaleWithImpliedRescales(Operation *op)
{
    if (implied_scales.find(op) != implied_scales.end()) {
        return implied_scales[op];

    } else {
        return -1;
    }
}

int ScaleAnalysis::getScaleWithoutImpliedRescales(Operation *op)
{
    if (real_scales.find(op) != real_scales.end()) {
        return real_scales[op];

    } else {
        return -1;
    }
}

int ScaleAnalysis::getValueScaleWithoutImpliedRescales(Value value)
{
    if (!value.getType().isa<CipherType>()) {
        return -1;
    }

    Operation *op = value.getDefiningOp();
    if (op) {
        return getScaleWithoutImpliedRescales(op);
    } else {
        return argument_scale;
    }
}


int ScaleAnalysis::recommendRescale(Operation *op)
{
    if (rescales_needed.find(op) != rescales_needed.end()) {
        return rescales_needed[op];

    } else {
        return -1;
    }
}

int ScaleAnalysis::calc_scale(Operation *op, bool implied_rescales)
{
    int result_scale = 0;

    std::string op_name = op->getName().getStringRef().str();

    if (op_name == "eva.constant") {
        return op->getAttr("result_scale").cast<IntegerAttr>().getSInt();
    }

    bool is_mul = op_name == "eva.multiply";

    for (auto value : op->getOperands())
    {
        Operation *d_op = value.getDefiningOp();
        int scale;

        if (d_op) {
            scale = implied_rescales ? getScaleWithImpliedRescales(d_op)
                    : getScaleWithoutImpliedRescales(d_op);
        } else {
            scale = argument_scale;
        }
        assert(scale != -1);
        
        if (is_mul) {
            result_scale += scale;
        } else {
            result_scale = std::max(result_scale, scale);
        }
    }

    return result_scale;
}

//===----------------------------------------------------------------------===//
// EVA Modulo Analysis
//===----------------------------------------------------------------------===//

int ModuloAnalysis::argument_modulo = 5;

ModuloAnalysis::ModuloAnalysis(Operation *op, AnalysisManager &am)
{
    std::vector<Operation *> worklist;
    for (auto &region : op->getRegions())
    {
        region.walk([&worklist](Operation *op) {
            /* process only EVA Dialect operations, to discard the top level operation */
            auto dial = op->getDialect();
            if (dial and dial->getNamespace() == EVADialect::getDialectNamespace())
            {
                worklist.push_back(op);
            }
        });
    }

    for (auto op : worklist) {
        int result_modulo = calc_modulo(op);
        result_moduli[op] = result_modulo;
    }

    // backward pass to propagate the modulo chain length to the undecided values

    std::reverse(worklist.begin(), worklist.end());
    std::map<Operation *, int> max_uscase_moduli;

    for (auto op : worklist)
    {
        int result_modulo = getModuloChainLength(op);
        
        if (result_modulo == -1) {
            result_moduli[op] = max_uscase_moduli[op];
        } else {
            backpropagate_modulo(op, max_uscase_moduli);
        }
    }
}

// calculate the modulo chain length
int ModuloAnalysis::calc_modulo(Operation *op)
{
    std::string op_name = op->getName().getStringRef().str();

    if (op_name == "eva.constant") {
        return op->getAttr("result_mod").cast<IntegerAttr>().getSInt();
    }

    int result_modulo = argument_modulo + 1;

    for (auto value : op->getOperands())
    {
        Operation *d_op = value.getDefiningOp();

        int modulo = d_op ? getModuloChainLength(d_op) : argument_modulo;

        if (modulo != -1) {
            result_modulo = std::min(result_modulo, modulo);
        }
    }

    if (result_modulo == argument_modulo + 1) {
        result_modulo = -1;
    }

    if (result_modulo != -1 and
        (op_name == "eva.rescale" or op_name == "eva.modswitch")) {
        result_modulo -= 1;
    }

    return result_modulo;
}

void ModuloAnalysis::backpropagate_modulo(Operation *op, std::map<Operation *, int> &max_uscase_moduli)
{
    std::string op_name = op->getName().getStringRef().str();

    int input_modulo = getModuloChainLength(op);

    if (op_name == "eva.modswtich" or op_name == "eva.rescale") {
        input_modulo += 1;
    }

    for (auto value : op->getOperands())
    {
        Operation *d_op = value.getDefiningOp();

        if (getModuloChainLength(d_op) == -1) {

            if (max_uscase_moduli.count(d_op) == 0) {
                max_uscase_moduli[d_op] = input_modulo;
            } else {
                max_uscase_moduli[d_op] = std::max(max_uscase_moduli[d_op], input_modulo);
            }
        }
    }
}

bool ModuloAnalysis::isInvalidated(const AnalysisManager::PreservedAnalyses &pa)
{
    return true;
}

int ModuloAnalysis::getModuloChainLength(Operation *op)
{
    if (result_moduli.find(op) != result_moduli.end()) {
        return result_moduli[op];

    } else {
        return -1;
    }
}

int ModuloAnalysis::getValueModuloChainLength(Value value)
{
    if (!value.getType().isa<CipherType>()) {
        return -1;
    }
    Operation *op = value.getDefiningOp();
    if (op) {
        return getModuloChainLength(op);
    } else {
        return argument_modulo;
    }
}