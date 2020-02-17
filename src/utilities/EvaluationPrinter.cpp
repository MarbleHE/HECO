#include "EvaluationPrinter.h"

void EvaluationPrinter::ensureEvalParamsAreSet() {
    if (evaluationParameters == nullptr) {
        throw std::logic_error(
                "EvaluationPrinter requires calling setEvaluationParameters(...) to have access to the passed parameters.");
    }
}

EvaluationPrinter::EvaluationPrinter() : evaluationParameters(nullptr) {}

EvaluationPrinter &EvaluationPrinter::setFlagPrintEachParameterSet(bool printEachParameterSet) {
    EvaluationPrinter::flagPrintEachParameterSet = printEachParameterSet;
    return *this;
}

EvaluationPrinter &EvaluationPrinter::setFlagPrintVariableHeaderOnceOnly(bool printVariableHeaderOnceOnly) {
    EvaluationPrinter::flagPrintVariableHeaderOnceOnly = printVariableHeaderOnceOnly;
    return *this;
}

EvaluationPrinter &EvaluationPrinter::setEvaluationParameters(std::unordered_map<std::string, Literal *> *evalParams) {
    EvaluationPrinter::evaluationParameters = evalParams;
    return *this;
}

EvaluationPrinter &EvaluationPrinter::setFlagPrintEvaluationResult(bool printEvaluationResult) {
    EvaluationPrinter::flagPrintEvaluationResult = printEvaluationResult;
    return *this;
}

void EvaluationPrinter::printHeader() {
    if (flagPrintEachParameterSet || flagPrintEvaluationResult) {
        if (flagPrintEachParameterSet) {
            ensureEvalParamsAreSet();
            for (auto &[varIdentifier, literal] : *evaluationParameters) std::cout << varIdentifier << ", ";
            std::cout << "\b\b" << std::endl;
        }
        if (flagPrintEvaluationResult) {
            std::cout << "Evaluation Result (original / rewritten)" << std::endl;
        }
        std::cout << std::endl;
    }
}

void EvaluationPrinter::printCurrentParameterSet() {
    if (flagPrintEachParameterSet) {
        ensureEvalParamsAreSet();
        std::stringstream varIdentifiers, varValues;
        for (auto &[varIdentifier, literal] : *evaluationParameters) {
            varIdentifiers << varIdentifier << ", ";
            varValues << *literal << ", ";
        }
        varIdentifiers << "\b\b" << std::endl;
        varValues << "\b\b" << std::endl;
        std::cout << (!flagPrintVariableHeaderOnceOnly ? varIdentifiers.str() : "") << varValues.str();
    }
}

void EvaluationPrinter::printEvaluationResults(const std::vector<Literal *> &resultExpected,
                                               const std::vector<Literal *> &resultRewrittenAst) {
    if (flagPrintEvaluationResult) {
        std::cout << "( ";
        for (auto &result : resultExpected) std::cout << result << ", ";
        std::cout << "\b\b";
        std::cout << " / ";
        for (auto &result : resultRewrittenAst) std::cout << result << ", ";
        std::cout << "\b\b";
        std::cout << " )";
    }
}

void EvaluationPrinter::printEndOfEvaluationTestRun() {
    std::cout << std::endl;
}
