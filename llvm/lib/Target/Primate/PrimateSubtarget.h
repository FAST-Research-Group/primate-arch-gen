//===-- PrimateSubtarget.h - Define Subtarget for the Primate -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Primate specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PRIMATE_PRIMATESUBTARGET_H
#define LLVM_LIB_TARGET_PRIMATE_PRIMATESUBTARGET_H

#include "MCTargetDesc/PrimateBaseInfo.h"
#include "PrimateFrameLowering.h"
#include "PrimateISelLowering.h"
#include "PrimateInstrInfo.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#define GET_SUBTARGETINFO_HEADER
#include "PrimateGenSubtargetInfo.inc"

namespace llvm {
class StringRef;

class PrimateSubtarget : public PrimateGenSubtargetInfo {
  uint8_t MaxInterleaveFactor = 2;
  PrimateABI::ABI TargetABI = PrimateABI::ABI_Unknown;
  BitVector UserReservedRegister;
  PrimateFrameLowering FrameLowering;
  PrimateInstrInfo InstrInfo;
  PrimateRegisterInfo RegInfo;
  PrimateTargetLowering TLInfo;
  SelectionDAGTargetInfo TSInfo;
  InstrItineraryData InstrItins;

  /// Initializes using the passed in CPU and feature strings so that we can
  /// use initializer lists for subtarget initialization.
  PrimateSubtarget &initializeSubtargetDependencies(const Triple &TT,
                                                  StringRef CPU,
                                                  StringRef TuneCPU,
                                                  StringRef FS,
                                                  StringRef ABIName);

public:
  // Initializes the data members to match that of the specified triple.
  PrimateSubtarget(const Triple &TT, StringRef CPU, StringRef TuneCPU,
                 StringRef FS, StringRef ABIName, const TargetMachine &TM);

  // Parses features string setting specified subtarget options. The
  // definition of this function is auto-generated by tblgen.
  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  const PrimateFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const InstrItineraryData *getInstrItineraryData() const override {
    return &InstrItins;
  }
  const PrimateInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const PrimateRegisterInfo *getRegisterInfo() const override {
    return &RegInfo;
  }
  const PrimateTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }
  bool enableMachineScheduler() const override { return true; }
  PrimateABI::ABI getTargetABI() const { return TargetABI; }
  bool isRegisterReservedByUser(Register i) const {
    assert(i < Primate::NUM_TARGET_REGS && "Register out of range");
    return UserReservedRegister[i];
  }
  unsigned getMaxInterleaveFactor() const {
    return 1;
  }

  virtual void anchor();

#define GET_SUBTARGETINFO_MACRO(ATTRIBUTE, DEFAULT, GETTER) \
  bool ATTRIBUTE = DEFAULT;
#include "PrimateGenSubtargetInfo.inc"

  unsigned ZvlLen = 0;
  bool is64Bit() const { return IsPR64; }
  MVT getXLenVT() const {
    return is64Bit() ? MVT::i64 : MVT::i32;
  }
  unsigned getXLen() const {
    return is64Bit() ? 64 : 32;
  }


protected:
  // GlobalISel related APIs.
  std::unique_ptr<CallLowering> CallLoweringInfo;
  std::unique_ptr<InstructionSelector> InstSelector;
  std::unique_ptr<LegalizerInfo> Legalizer;
  std::unique_ptr<RegisterBankInfo> RegBankInfo;

public:
  const CallLowering *getCallLowering() const override;
  InstructionSelector *getInstructionSelector() const override;
  const LegalizerInfo *getLegalizerInfo() const override;
  const RegisterBankInfo *getRegBankInfo() const override;

#define GET_SUBTARGETINFO_MACRO(ATTRIBUTE, DEFAULT, GETTER) \
  bool GETTER() const { return ATTRIBUTE; }
#include "PrimateGenSubtargetInfo.inc"

  bool hasStdExtCOrZca() const { return HasStdExtC || HasStdExtZca; }
  bool hasStdExtZvl() const { return ZvlLen != 0; }
  bool hasStdExtFOrZfinx() const { return HasStdExtF || HasStdExtZfinx; }
  bool hasStdExtDOrZdinx() const { return HasStdExtD || HasStdExtZdinx; }
  bool hasStdExtZfhOrZhinx() const { return HasStdExtZfh || HasStdExtZhinx; }
  bool hasStdExtZfhminOrZhinxmin() const {
    return HasStdExtZfhmin || HasStdExtZhinxmin;
  }
  bool hasHalfFPLoadStoreMove() const {
    return HasStdExtZfhmin || HasStdExtZfbfmin;
  }

  // Return the known range for the bit length of PRV data registers. A value
  // of 0 means nothing is known about that particular limit beyond what's
  // implied by the architecture.
  unsigned getMaxPRVVectorSizeInBits() const;
  unsigned getMinPRVVectorSizeInBits() const;
  unsigned getMaxLMULForFixedLengthVectors() const;
  bool usePRVForFixedLengthVectors() const;
};
} // End llvm namespace

#endif
