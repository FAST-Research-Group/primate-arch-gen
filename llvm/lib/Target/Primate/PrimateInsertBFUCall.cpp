#include "PrimateInsertBFUCall.h"
// #include "llvm/CodeGen/GlobalISel/Combiner.h"
// #include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
// #include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

// Look at AMDGPUPreLegalizerCombiner
// class AMDGPUPreLegalizerCombinerImpl : public Combiner {
// protected:
//   const AMDGPUPreLegalizerCombinerImplRuleConfig &RuleConfig;
//   const GCNSubtarget &STI;

using namespace llvm;

#define DEBUG_TYPE "primate-isel"
#define PASS_NAME "Primate DAG->DAG Pattern Instruction Selection"


bool PrimateInsertBFUCall::runOnMachineFunction(MachineFunction &MF) {
  // CurDAG->init(*MF, *ORE, this, LibInfo, UA, PSI, BFI, FnVarLocs);
  
  llvm::dbgs() << "=====================================\n";
  llvm::dbgs() << "Hello from Primate BFU Call Insertion\n\n";

  llvm::dbgs() << "Dump Machine Function:\n";
  MF.dump();

  llvm::dbgs() << "Dump CurDAG:\n";
  CurDAG->dump();

  llvm::dbgs() << "Dump DAG Nodes:\n";
  for (SelectionDAG::allnodes_iterator I = CurDAG->allnodes_begin(),
                                       E = CurDAG->allnodes_end(); I != E;) {
    SDNode *N = &*I++; // Preincrement iterator to avoid invalidation issues.
    N->dump();
  }

  llvm::dbgs() << "\nGoodbye from Primate BFU Call Insertion\n";
  llvm::dbgs() << "=====================================\n";

  return false;
}

void PrimateInsertBFUCall::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
}

char PrimateInsertBFUCall::ID = 0;

