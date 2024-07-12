//===-- PrimateInstrInfo.h - Primate Instruction Information --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Primate implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PRIMATE_PRIMATEINSTRINFO_H
#define LLVM_LIB_TARGET_PRIMATE_PRIMATEINSTRINFO_H

#include "PrimateRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DiagnosticInfo.h"

#define GET_INSTRINFO_HEADER
#define GET_INSTRINFO_OPERAND_ENUM
#include "PrimateGenInstrInfo.inc"

namespace llvm {

class PrimateSubtarget;

static const MachineMemOperand::Flags MONontemporalBit0 =
    MachineMemOperand::MOTargetFlag1;
static const MachineMemOperand::Flags MONontemporalBit1 =
    MachineMemOperand::MOTargetFlag2;

namespace PrimateCC {

enum CondCode {
  COND_EQ,
  COND_NE,
  COND_LT,
  COND_GE,
  COND_LTU,
  COND_GEU,
  COND_INVALID
};

CondCode getOppositeBranchCondition(CondCode);
unsigned getBrCond(CondCode CC);

} // end of namespace PrimateCC
class PrimateInstrInfo : public PrimateGenInstrInfo {

public:
  explicit PrimateInstrInfo(PrimateSubtarget &STI);

  MCInst getNop() const override;

  unsigned isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                   const DebugLoc &DL, MCRegister DstReg, MCRegister SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI,
                           Register SrcReg, bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI,
                           Register VReg) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            Register DestReg, int FrameIndex,
                            const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI,
                            Register VReg) const override;
  
  outliner::InstrType getOutliningType(MachineBasicBlock::iterator &MBBI, 
                                       unsigned Flags) const;

  // Materializes the given integer Val into DstReg.
  void movImm(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
              const DebugLoc &DL, Register DstReg, uint64_t Val,
              MachineInstr::MIFlag Flag = MachineInstr::NoFlags) const;

  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify) const override;

  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &dl,
                        int *BytesAdded = nullptr) const override;

  void insertIndirectBranch(MachineBasicBlock &MBB,
                            MachineBasicBlock &NewDestBB,
                            MachineBasicBlock &RestoreBB,
                            const DebugLoc &DL, int64_t BrOffset = 0,
                            RegScavenger *RS = nullptr) const override;

  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;

  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;

  MachineBasicBlock *getBranchDestBlock(const MachineInstr &MI) const override;

  bool isBranchOffsetInRange(unsigned BranchOpc,
                             int64_t BrOffset) const override;

  bool isAsCheapAsAMove(const MachineInstr &MI) const override;

  std::optional<DestSourcePair>
  isCopyInstrImpl(const MachineInstr &MI) const override;

  bool verifyInstruction(const MachineInstr &MI,
                         StringRef &ErrInfo) const override;

  bool getMemOperandWithOffsetWidth(const MachineInstr &LdSt,
                                    const MachineOperand *&BaseOp,
                                    int64_t &Offset, unsigned &Width,
                                    const TargetRegisterInfo *TRI) const;

  DFAPacketizer *
  CreateTargetScheduleState(const TargetSubtargetInfo &STI) const override;

  bool areMemAccessesTriviallyDisjoint(const MachineInstr &MIa,
                                       const MachineInstr &MIb) const override;


  std::pair<unsigned, unsigned>
  decomposeMachineOperandsTargetFlags(unsigned TF) const override;

  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableDirectMachineOperandTargetFlags() const override;

  // Return true if the function can safely be outlined from.
  virtual bool
  isFunctionSafeToOutlineFrom(MachineFunction &MF,
                              bool OutlineFromLinkOnceODRs) const override;

  // Return true if MBB is safe to outline from, and return any target-specific
  // information in Flags.
  virtual bool isMBBSafeToOutlineFrom(MachineBasicBlock &MBB,
                                      unsigned &Flags) const override;

  // Calculate target-specific information for a set of outlining candidates.
  std::optional<outliner::OutlinedFunction> getOutliningCandidateInfo(
      std::vector<outliner::Candidate> &RepeatedSequenceLocs) const override;

  // Insert a custom frame for outlined functions.
  virtual void
  buildOutlinedFrame(MachineBasicBlock &MBB, MachineFunction &MF,
                     const outliner::OutlinedFunction &OF) const override;
  
  MachineBasicBlock::iterator insertOutlinedCall(
      Module &M, MachineBasicBlock &MBB, MachineBasicBlock::iterator &It,
      MachineFunction &MF, outliner::Candidate &C) const override;

  bool findCommutedOpIndices(const MachineInstr &MI, unsigned &SrcOpIdx1,
                             unsigned &SrcOpIdx2) const override;
  MachineInstr *commuteInstructionImpl(MachineInstr &MI, bool NewMI,
                                       unsigned OpIdx1,
                                       unsigned OpIdx2) const override;

  MachineInstr *convertToThreeAddress(MachineInstr &MI,
                                      LiveVariables *LV,
                                      LiveIntervals *LIS) const override;

  Register getVLENFactoredAmount(
      MachineFunction &MF, MachineBasicBlock &MBB,
      MachineBasicBlock::iterator II, const DebugLoc &DL, int64_t Amount,
      MachineInstr::MIFlag Flag = MachineInstr::NoFlags) const;

  // Returns true if the given MI is an PRV instruction opcode for which we may
  // expect to see a FrameIndex operand. When CheckFIs is true, the instruction
  // must contain at least one FrameIndex operand.
  bool isPRVSpill(const MachineInstr &MI, bool CheckFIs) const;

  std::optional<std::pair<unsigned, unsigned>>
  isPRVSpillForZvlsseg(unsigned Opcode) const;

protected:
  const PrimateSubtarget &STI;
};

namespace PrimateVPseudosTable {

struct PseudoInfo {
  uint16_t Pseudo;
  uint16_t BaseInstr;
};

#define GET_PrimateVPseudosTable_DECL
#include "PrimateGenSearchableTables.inc"

} // end namespace PrimateVPseudosTable

} // end namespace llvm
#endif
