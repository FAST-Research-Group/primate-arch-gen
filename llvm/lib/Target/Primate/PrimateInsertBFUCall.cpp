#include "PrimateInsertBFUCall.h"
// #include "llvm/CodeGen/GlobalISel/Combiner.h"
// #include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
// #include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Pass.h"

// Look at AMDGPUPreLegalizerCombiner
// class AMDGPUPreLegalizerCombinerImpl : public Combiner {
// protected:
//   const AMDGPUPreLegalizerCombinerImplRuleConfig &RuleConfig;
//   const GCNSubtarget &STI;

using namespace llvm;

#define DEBUG_TYPE "primate-isel"
#define PASS_NAME "Primate DAG->DAG Pattern Instruction Selection"


// This pass transforms a Primate-specific DAG by converting matching 
// instruction blocks into BFU call instructions
// FunctionPass *llvm::createPrimateInsertBFUCall(PrimateTargetMachine &TM) {
// MachineFunctionPass *llvm::createPrimateInsertBFUCall() {
//   // return new PrimateInsertBFUCall(TM);
//   return new PrimateInsertBFUCall();
// }

char PrimateInsertBFUCall::ID = 0;

