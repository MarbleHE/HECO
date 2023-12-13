#ifndef HECO_IR_EVA_EVAANALYSES_H
#define HECO_IR_EVA_EVAANALYSES_H

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/IR/Operation.h"
#include "iostream"

using namespace mlir;

namespace heco
{
    namespace eva
    {
        class ScaleAnalysis
        {
        public:
            ScaleAnalysis(Operation *op, AnalysisManager &am);

            bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa);
            int getScaleWithoutImpliedRescales(Operation *op);
            int getScaleWithImpliedRescales(Operation *op);
            int getValueScaleWithoutImpliedRescales(Value value);

            int recommendRescale(Operation *op);

            static int argument_scale;
            static int waterline;
            static int scale_drop;

        private:
            std::map<Operation *, int> rescales_needed;
            /* result scales without implied rescales */
            std::map<Operation *, int> real_scales;
            /* result scales with implied rescales */
            std::map<Operation *, int> implied_scales;

            int calc_scale(Operation *op, bool implied_rescales);
        };

        class ModuloAnalysis
        {
        public:
            ModuloAnalysis(Operation *op, AnalysisManager &am);

            bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa);
            int getModuloChainLength(Operation *op);
            int getValueModuloChainLength(Value value);

            static int argument_modulo;

        private:
            std::map<Operation *, int> result_moduli;

            int calc_modulo(Operation *op);
            void backpropagate_modulo(Operation *op, std::map<Operation *, int> &max_usecase_moduli);
        };
    }
}

#endif //HECO_IR_EVA_EVAANALYSES_H