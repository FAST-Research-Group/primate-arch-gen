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
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

// namespace {

class PrimateInsertBFUCall : public MachineFunctionPass {
  // const PrimateSubtarget *Subtarget = nullptr;

public:
  static char ID;

  // explicit 
  PrimateInsertBFUCall() //(PrimateTargetMachine &TargetMachine)
      : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Primate BFU Call Insertion";
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    llvm::dbgs() << "Hello from Primate BFU Call Insertion\n";
    // Subtarget = &MF.getSubtarget<PrimateSubtarget>();
    // return MachineFunctionPass::runOnMachineFunction(MF);
    return false;
  }
  
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};


// } // end anonymous namespace

// char PrimateInsertBFUCall::ID = 0;

} // namespace llvm

#endif
