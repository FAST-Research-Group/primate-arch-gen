//===---- PrimateInsertBFUCall.h - Matches inst blocks to BFU calls -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Something something, BFU sub-graph matching, isomorphism whatever
// We need this to run during instruction selection, but AFTER legalization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PRIMATE_PRIMATEINSERTBFUCALL_H
#define LLVM_LIB_TARGET_PRIMATE_PRIMATEINSERTBFUCALL_H

#include "Primate.h"
#include "PrimateTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SwiftErrorValueTracking.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class PrimateInsertBFUCall : public MachineFunctionPass {
public:
  static char ID;

  // TargetMachine &TM;
  PrimateTargetMachine &TM;
  // const TargetLibraryInfo *LibInfo;
  // std::unique_ptr<FunctionLoweringInfo> FuncInfo;
  // SwiftErrorValueTracking *SwiftError;
  // // MachineFunction *MF;
  // MachineRegisterInfo *RegInfo;
  SelectionDAG *CurDAG;
  // std::unique_ptr<SelectionDAGBuilder> SDB;
  // AAResults *AA = nullptr;
  // AssumptionCache *AC = nullptr;
  // GCFunctionInfo *GFI = nullptr;
  // CodeGenOptLevel OptLevel;
  // const TargetInstrInfo *TII;
  // const TargetLowering *TLI;
  // bool FastISelFailed;
  // SmallPtrSet<const Instruction *, 4> ElidedArgCopyInstrs;

  explicit PrimateInsertBFUCall(PrimateTargetMachine &TargetMachine, 
                                CodeGenOptLevel OL)
      : MachineFunctionPass(ID), 
        TM(TargetMachine),
        CurDAG(new SelectionDAG(TargetMachine, OL)) //,
//       SDB(std::make_unique<SelectionDAGBuilder>(*CurDAG, *FuncInfo, *SwiftError,
//                                                 OL)),
//       OptLevel(OL) {
//   initializeGCModuleInfoPass(*PassRegistry::getPassRegistry());
//   initializeBranchProbabilityInfoWrapperPassPass(
//       *PassRegistry::getPassRegistry());
//   initializeAAResultsWrapperPassPass(*PassRegistry::getPassRegistry());
//   initializeTargetLibraryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
// }
  {}


  StringRef getPassName() const override {
    return "Primate BFU Call Insertion";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace llvm

#endif
