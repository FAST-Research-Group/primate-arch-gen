//===-- PrimateISelDAGToDAG.cpp - A dag to dag inst selector for Primate ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the Primate target.
//
//===----------------------------------------------------------------------===//

#include "PrimateISelDAGToDAG.h"
#include "MCTargetDesc/PrimateMCTargetDesc.h"
#include "MCTargetDesc/PrimateMatInt.h"
#include "PrimateISelLowering.h"
#include "PrimateMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/IntrinsicsPrimate.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "primate-isel"
#define PASS_NAME "Primate DAG->DAG Pattern Instruction Selection"



namespace llvm {
namespace Primate {
#define GET_PrimateVSSEGTable_IMPL
#define GET_PrimateVLSEGTable_IMPL
#define GET_PrimateVLXSEGTable_IMPL
#define GET_PrimateVSXSEGTable_IMPL
#define GET_PrimateVLETable_IMPL
#define GET_PrimateVSETable_IMPL
#define GET_PrimateVLXTable_IMPL
#define GET_PrimateVSXTable_IMPL
#include "PrimateGenSearchableTables.inc"
} // namespace Primate
} // namespace llvm

void PrimateDAGToDAGISel::PreprocessISelDAG() {
  dbgs() << "Here in PreProcessISelDAG\n";
  CurDAG->AssignTopologicalOrder();
  CurDAG->dump();
  
  for (SDNode &N : CurDAG->allnodes()) {
    SDLoc DL(&N);
    if (N.getOpcode() == PrimateISD::EXTRACT || N.getOpcode() == PrimateISD::INSERT ||
        N.getOpcode() == ISD::Constant) {
      continue;
    }

    getSubgraph(&N, DL);
    // for(unsigned i = 0; i < N.getNumValues(); i++ ) {
    //   SDValue CurVal = SDValue(&N, i);
    //   if(CurVal.getValueType() == MVT::i32) {
    //     dbgs() << "Found an instr that has an i32 type :D\n";
    //     N.dump();
        // SmallVector<SDValue> OPs {CurVal, CurDAG->getTargetConstant(0, DL, MVT::i32)};
    //     for (const auto &OP : OPs)
    //       OP->dump();
    //     // CurDAG->getNode(Primate::INSERTdef, DL, MVT::Primate_aggregate, OPs);
    //     dbgs() << "\n";
    //   }
    // }
  }
}

/// Look for various patterns that can be done with a SHL that can be folded
/// into a SHXADD_UW. \p ShAmt contains 1, 2, or 3 and is set based on which
/// SHXADD_UW we are trying to match.
bool PrimateDAGToDAGISel::selectSHXADD_UWOp(SDValue N, unsigned ShAmt,
                                          SDValue &Val) {
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1)) &&
      N.hasOneUse()) {
    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N0.hasOneUse()) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      Mask &= maskTrailingZeros<uint64_t>(C2);

      // Look for (and (shl y, c2), c1) where c1 is a shifted mask with
      // 32-ShAmt leading zeros and c2 trailing zeros. We can use SLLI by
      // c2-ShAmt followed by SHXADD_UW with ShAmt for the X amount.
      if (isShiftedMask_64(Mask)) {
        unsigned Leading = llvm::countl_zero(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        if (Leading == 32 - ShAmt && Trailing == C2 && Trailing > ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Primate::SLLI, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(C2 - ShAmt, DL, VT)),
                        0);
          return true;
        }
      }
    }
  }

  return false;
}

void PrimateDAGToDAGISel::PostprocessISelDAG() {
  dbgs() << "Here in PrimateDAGToDAGISel::PostprocessISelDAG()\n";

  CurDAG->dump();

  doPeepholeLoadStoreADDI();
}

static SDNode *selectImm(SelectionDAG *CurDAG, const SDLoc &DL, int64_t Imm,
                         const PrimateSubtarget &Subtarget) {
  MVT XLenVT = Subtarget.getXLenVT();
  PrimateMatInt::InstSeq Seq =
      PrimateMatInt::generateInstSeq(Imm, Subtarget);

  SDNode *Result = nullptr;
  SDValue SrcReg = CurDAG->getRegister(Primate::X0, XLenVT);
  for (PrimateMatInt::Inst &Inst : Seq) {
    SDValue SDImm = CurDAG->getTargetConstant(Inst.getImm(), DL, XLenVT);
    if (Inst.getOpcode() == Primate::LUI)
      Result = CurDAG->getMachineNode(Primate::LUI, DL, XLenVT, SDImm);
    else if (Inst.getOpcode() == Primate::ADD_UW)
      Result = CurDAG->getMachineNode(Primate::ADD_UW, DL, XLenVT, SrcReg,
                                      CurDAG->getRegister(Primate::X0, XLenVT));
    else
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, XLenVT, SrcReg, SDImm);

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return Result;
}

static SDValue createTupleImpl(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                               unsigned RegClassID, unsigned SubReg0) {
  assert(Regs.size() >= 2 && Regs.size() <= 8);

  SDLoc DL(Regs[0]);
  SmallVector<SDValue, 8> Ops;

  Ops.push_back(CurDAG.getTargetConstant(RegClassID, DL, MVT::i32));

  for (unsigned I = 0; I < Regs.size(); ++I) {
    Ops.push_back(Regs[I]);
    Ops.push_back(CurDAG.getTargetConstant(SubReg0 + I, DL, MVT::i32));
  }
  SDNode *N =
      CurDAG.getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::Untyped, Ops);
  return SDValue(N, 0);
}

static SDValue createM1Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  static const unsigned RegClassIDs[] = {
      Primate::VRN2M1RegClassID, Primate::VRN3M1RegClassID, Primate::VRN4M1RegClassID,
      Primate::VRN5M1RegClassID, Primate::VRN6M1RegClassID, Primate::VRN7M1RegClassID,
      Primate::VRN8M1RegClassID};

  return createTupleImpl(CurDAG, Regs, RegClassIDs[NF - 2], Primate::sub_vrm1_0);
}

static SDValue createM2Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  static const unsigned RegClassIDs[] = {Primate::VRN2M2RegClassID,
                                         Primate::VRN3M2RegClassID,
                                         Primate::VRN4M2RegClassID};

  return createTupleImpl(CurDAG, Regs, RegClassIDs[NF - 2], Primate::sub_vrm2_0);
}

static SDValue createM4Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  return createTupleImpl(CurDAG, Regs, Primate::VRN2M4RegClassID,
                         Primate::sub_vrm4_0);
}

static SDValue createTuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                           unsigned NF, PrimateII::VLMUL LMUL) {
  switch (LMUL) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case PrimateII::VLMUL::LMUL_F8:
  case PrimateII::VLMUL::LMUL_F4:
  case PrimateII::VLMUL::LMUL_F2:
  case PrimateII::VLMUL::LMUL_1:
    return createM1Tuple(CurDAG, Regs, NF);
  case PrimateII::VLMUL::LMUL_2:
    return createM2Tuple(CurDAG, Regs, NF);
  case PrimateII::VLMUL::LMUL_4:
    return createM4Tuple(CurDAG, Regs, NF);
  }
}

void PrimateDAGToDAGISel::addVectorLoadStoreOperands(
    SDNode *Node, unsigned Log2SEW, const SDLoc &DL, unsigned CurOp,
    bool IsMasked, bool IsStridedOrIndexed, SmallVectorImpl<SDValue> &Operands,
    MVT *IndexVT) {
  SDValue Chain = Node->getOperand(0);
  SDValue Glue;

  SDValue Base;
  SelectBaseAddr(Node->getOperand(CurOp++), Base);
  Operands.push_back(Base); // Base pointer.

  if (IsStridedOrIndexed) {
    Operands.push_back(Node->getOperand(CurOp++)); // Index.
    if (IndexVT)
      *IndexVT = Operands.back()->getSimpleValueType(0);
  }

  if (IsMasked) {
    // Mask needs to be copied to V0.
    SDValue Mask = Node->getOperand(CurOp++);
    Chain = CurDAG->getCopyToReg(Chain, DL, Primate::V0, Mask, SDValue());
    Glue = Chain.getValue(1);
    Operands.push_back(CurDAG->getRegister(Primate::V0, Mask.getValueType()));
  }
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);

  MVT XLenVT = Subtarget->getXLenVT();
  SDValue SEWOp = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);
  Operands.push_back(SEWOp);

  Operands.push_back(Chain); // Chain.
  if (Glue)
    Operands.push_back(Glue);
}

void PrimateDAGToDAGISel::selectVLSEG(SDNode *Node, bool IsMasked,
                                    bool IsStrided) {
  llvm_unreachable("should never see VLSeg inst");
}

void PrimateDAGToDAGISel::selectVLSEGFF(SDNode *Node, bool IsMasked) {
  llvm_unreachable("should never see VLSEGFF inst in primate");
}

void PrimateDAGToDAGISel::selectVLXSEG(SDNode *Node, bool IsMasked,
                                     bool IsOrdered) {
  llvm_unreachable("should never see VLSEG inst in primate");
}

void PrimateDAGToDAGISel::selectVSSEG(SDNode *Node, bool IsMasked,
                                    bool IsStrided) {
  llvm_unreachable("should never see VSSEG inst in primate");
}

void PrimateDAGToDAGISel::selectVSXSEG(SDNode *Node, bool IsMasked,
                                     bool IsOrdered) {
  llvm_unreachable("should never see VSXSEG inst in primate");
}


void PrimateDAGToDAGISel::Select(SDNode *Node) {
  // If we have a custom node, we have already selected.
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  dbgs() << "Selecting nodes for: ";
  Node->dump();
  dbgs() << "\n";

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  unsigned Opcode = Node->getOpcode();
  MVT XLenVT = Subtarget->getXLenVT();
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);

  switch (Opcode) {
    case ISD::Constant: {
      auto *ConstNode = cast<ConstantSDNode>(Node);
      if (VT == XLenVT && ConstNode->isZero()) {
        SDValue New =
            CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL, Primate::X0, XLenVT);
        ReplaceNode(Node, New.getNode());
        return;
      }
      ReplaceNode(Node,
                  selectImm(CurDAG, DL, ConstNode->getSExtValue(), *Subtarget));
      return;
    }
    case ISD::FrameIndex: {
      SDValue Imm = CurDAG->getTargetConstant(0, DL, XLenVT);
      int FI = cast<FrameIndexSDNode>(Node)->getIndex();
      SDValue TFI = CurDAG->getTargetFrameIndex(FI, VT);
      ReplaceNode(Node, CurDAG->getMachineNode(Primate::ADDI, DL, VT, TFI, Imm));
      return;
    }
    case ISD::SRL: {
      // We don't need this transform if zext.h is supported.
      if (Subtarget->hasStdExtZbb())
        break;
      // Optimize (srl (and X, 0xffff), C) ->
      //          (srli (slli X, (XLen-16), (XLen-16) + C)
      // Taking into account that the 0xffff may have had lower bits unset by
      // SimplifyDemandedBits. This avoids materializing the 0xffff immediate.
      // This pattern occurs when type legalizing i16 right shifts.
      // FIXME: This could be extended to other AND masks.
      auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
      if (N1C) {
        uint64_t ShAmt = N1C->getZExtValue();
        SDValue N0 = Node->getOperand(0);
        if (ShAmt < 16 && N0.getOpcode() == ISD::AND && N0.hasOneUse() &&
            isa<ConstantSDNode>(N0.getOperand(1))) {
          uint64_t Mask = N0.getConstantOperandVal(1);
          Mask |= maskTrailingOnes<uint64_t>(ShAmt);
          if (Mask == 0xffff) {
            unsigned LShAmt = Subtarget->getXLen() - 16;
            SDNode *SLLI =
                CurDAG->getMachineNode(Primate::SLLI, DL, VT, N0->getOperand(0),
                                       CurDAG->getTargetConstant(LShAmt, DL, VT));
            SDNode *SRLI = CurDAG->getMachineNode(
                Primate::SRLI, DL, VT, SDValue(SLLI, 0),
                CurDAG->getTargetConstant(LShAmt + ShAmt, DL, VT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      break;
    }
    case ISD::AND: {
      auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
      if (!N1C)
        break;

      SDValue N0 = Node->getOperand(0);

      bool LeftShift = N0.getOpcode() == ISD::SHL;
      if (!LeftShift && N0.getOpcode() != ISD::SRL)
        break;

      auto *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C)
        break;
      uint64_t C2 = C->getZExtValue();
      unsigned XLen = Subtarget->getXLen();
      if (!C2 || C2 >= XLen)
        break;

      uint64_t C1 = N1C->getZExtValue();

      // Keep track of whether this is a andi, zext.h, or zext.w.
      bool ZExtOrANDI = isInt<12>(N1C->getSExtValue());
      if (C1 == UINT64_C(0xFFFF) &&
          (Subtarget->hasStdExtZbb()))
        ZExtOrANDI = true;
      if (C1 == UINT64_C(0xFFFFFFFF) && Subtarget->hasStdExtZba())
        ZExtOrANDI = true;

      // Clear irrelevant bits in the mask.
      if (LeftShift)
        C1 &= maskTrailingZeros<uint64_t>(C2);
      else
        C1 &= maskTrailingOnes<uint64_t>(XLen - C2);

      // Some transforms should only be done if the shift has a single use or
      // the AND would become (srli (slli X, 32), 32)
      bool OneUseOrZExtW = N0.hasOneUse() || C1 == UINT64_C(0xFFFFFFFF);

      SDValue X = N0.getOperand(0);

      // Turn (and (srl x, c2) c1) -> (srli (slli x, c3-c2), c3) if c1 is a mask
      // with c3 leading zeros.
      if (!LeftShift && isMask_64(C1)) {
        uint64_t C3 = XLen - (llvm::bit_width(C1));
        if (C2 < C3) {
          // If the number of leading zeros is C2+32 this can be SRLIW.
          if (C2 + 32 == C3) {
            SDNode *SRLIW =
                CurDAG->getMachineNode(Primate::SRLIW, DL, XLenVT, X,
                                       CurDAG->getTargetConstant(C2, DL, XLenVT));
            ReplaceNode(Node, SRLIW);
            return;
          }

          // (and (srl (sexti32 Y), c2), c1) -> (srliw (sraiw Y, 31), c3 - 32) if
          // c1 is a mask with c3 leading zeros and c2 >= 32 and c3-c2==1.
          //
          // This pattern occurs when (i32 (srl (sra 31), c3 - 32)) is type
          // legalized and goes through DAG combine.
          SDValue Y;
          if (C2 >= 32 && (C3 - C2) == 1 && N0.hasOneUse() &&
              selectSExti32(X, Y)) {
            SDNode *SRAIW =
                CurDAG->getMachineNode(Primate::SRAIW, DL, XLenVT, Y,
                                       CurDAG->getTargetConstant(31, DL, XLenVT));
            SDNode *SRLIW = CurDAG->getMachineNode(
                Primate::SRLIW, DL, XLenVT, SDValue(SRAIW, 0),
                CurDAG->getTargetConstant(C3 - 32, DL, XLenVT));
            ReplaceNode(Node, SRLIW);
            return;
          }

          // (srli (slli x, c3-c2), c3).
          if (OneUseOrZExtW && !ZExtOrANDI) {
            SDNode *SLLI = CurDAG->getMachineNode(
                Primate::SLLI, DL, XLenVT, X,
                CurDAG->getTargetConstant(C3 - C2, DL, XLenVT));
            SDNode *SRLI =
                CurDAG->getMachineNode(Primate::SRLI, DL, XLenVT, SDValue(SLLI, 0),
                                       CurDAG->getTargetConstant(C3, DL, XLenVT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      // Turn (and (shl x, c2) c1) -> (srli (slli c2+c3), c3) if c1 is a mask
      // shifted by c2 bits with c3 leading zeros.
      if (LeftShift && isShiftedMask_64(C1)) {
        uint64_t C3 = XLen - (llvm::bit_width(C1));

        if (C2 + C3 < XLen &&
            C1 == (maskTrailingOnes<uint64_t>(XLen - (C2 + C3)) << C2)) {
          // Use slli.uw when possible.
          if ((XLen - (C2 + C3)) == 32 && Subtarget->hasStdExtZba()) {
            SDNode *SLLIUW =
                CurDAG->getMachineNode(Primate::SLLI_UW, DL, XLenVT, X,
                                       CurDAG->getTargetConstant(C2, DL, XLenVT));
            ReplaceNode(Node, SLLIUW);
            return;
          }

          // (srli (slli c2+c3), c3)
          if (OneUseOrZExtW && !ZExtOrANDI) {
            SDNode *SLLI = CurDAG->getMachineNode(
                Primate::SLLI, DL, XLenVT, X,
                CurDAG->getTargetConstant(C2 + C3, DL, XLenVT));
            SDNode *SRLI =
                CurDAG->getMachineNode(Primate::SRLI, DL, XLenVT, SDValue(SLLI, 0),
                                       CurDAG->getTargetConstant(C3, DL, XLenVT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      break;
    }
    case ISD::INTRINSIC_WO_CHAIN: {
      dbgs() << "Select for Intrinsic_wo_chain\n";

      unsigned IntNo = Node->getConstantOperandVal(0);
      switch (IntNo) {
        // By default we do not custom select any intrinsic.
        default:
          break;

        case Intrinsic::primate_vmsgeu:
        case Intrinsic::primate_vmsge: {
          SDValue Src1 = Node->getOperand(1);
          SDValue Src2 = Node->getOperand(2);
          // Only custom select scalar second operand.
          if (Src2.getValueType() != XLenVT)
            break;
          // Small constants are handled with patterns.
          if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
            int64_t CVal = C->getSExtValue();
            if (CVal >= -15 && CVal <= 16)
              break;
          }
          bool IsUnsigned = IntNo == Intrinsic::primate_vmsgeu;
          MVT Src1VT = Src1.getSimpleValueType();
          unsigned VMSLTOpcode, VMNANDOpcode;
          switch (PrimateTargetLowering::getLMUL(Src1VT)) {
            default:
              llvm_unreachable("Unexpected LMUL!");
            case PrimateII::VLMUL::LMUL_F8:
            case PrimateII::VLMUL::LMUL_F4:
            case PrimateII::VLMUL::LMUL_F2:
            case PrimateII::VLMUL::LMUL_1:
            case PrimateII::VLMUL::LMUL_2:
            case PrimateII::VLMUL::LMUL_4:
            case PrimateII::VLMUL::LMUL_8:
              llvm_unreachable("Primate should not generate VMSGE or VMSGEU Intrinsics");
          }
          SDValue SEW = CurDAG->getTargetConstant(
              Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
          SDValue VL;
          selectVLOp(Node->getOperand(3), VL);

          // Expand to
          // vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd
          SDValue Cmp = SDValue(
              CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
              0);
          ReplaceNode(Node, CurDAG->getMachineNode(VMNANDOpcode, DL, VT,
                                                   {Cmp, Cmp, VL, SEW}));
          return;
        }
        case Intrinsic::primate_vmsgeu_mask:
        case Intrinsic::primate_vmsge_mask: {
          SDValue Src1 = Node->getOperand(2);
          SDValue Src2 = Node->getOperand(3);
          // Only custom select scalar second operand.
          if (Src2.getValueType() != XLenVT)
            break;
          // Small constants are handled with patterns.
          if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
            int64_t CVal = C->getSExtValue();
            if (CVal >= -15 && CVal <= 16)
              break;
          }
          bool IsUnsigned = IntNo == Intrinsic::primate_vmsgeu_mask;
          MVT Src1VT = Src1.getSimpleValueType();
          unsigned VMSLTOpcode, VMSLTMaskOpcode, VMXOROpcode, VMANDNOTOpcode;
          switch (PrimateTargetLowering::getLMUL(Src1VT)) {
            default:
              llvm_unreachable("Unexpected LMUL!");
            case PrimateII::VLMUL::LMUL_F8:
            case PrimateII::VLMUL::LMUL_F4:
            case PrimateII::VLMUL::LMUL_F2:
            case PrimateII::VLMUL::LMUL_1:
            case PrimateII::VLMUL::LMUL_2:
            case PrimateII::VLMUL::LMUL_4:
            case PrimateII::VLMUL::LMUL_8:
              llvm_unreachable("Primate should not generate VMSGE{U}_MASK intrinsics");
          }
          // Mask operations use the LMUL from the mask type.
          switch (PrimateTargetLowering::getLMUL(VT)) {
            default:
              llvm_unreachable("Unexpected LMUL!");
            case PrimateII::VLMUL::LMUL_F8:
            case PrimateII::VLMUL::LMUL_F4:
            case PrimateII::VLMUL::LMUL_F2:
            case PrimateII::VLMUL::LMUL_1:
            case PrimateII::VLMUL::LMUL_2:
            case PrimateII::VLMUL::LMUL_4:
            case PrimateII::VLMUL::LMUL_8:
              llvm_unreachable("Primate should not generate VMSGE{U}_MASK intrinsics");
          }
          SDValue SEW = CurDAG->getTargetConstant(
              Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
          SDValue MaskSEW = CurDAG->getTargetConstant(0, DL, XLenVT);
          SDValue VL;
          selectVLOp(Node->getOperand(5), VL);
          SDValue MaskedOff = Node->getOperand(1);
          SDValue Mask = Node->getOperand(4);
          // If the MaskedOff value and the Mask are the same value use
          // vmslt{u}.vx vt, va, x;  vmandnot.mm vd, vd, vt
          // This avoids needing to copy v0 to vd before starting the next sequence.
          if (Mask == MaskedOff) {
            SDValue Cmp = SDValue(
                CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
                0);
            ReplaceNode(Node, CurDAG->getMachineNode(VMANDNOTOpcode, DL, VT,
                                                     {Mask, Cmp, VL, MaskSEW}));
            return;
          }

          // Mask needs to be copied to V0.
          SDValue Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), DL,
                                               Primate::V0, Mask, SDValue());
          SDValue Glue = Chain.getValue(1);
          SDValue V0 = CurDAG->getRegister(Primate::V0, VT);

          // Otherwise use
          // vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0
          SDValue Cmp = SDValue(
              CurDAG->getMachineNode(VMSLTMaskOpcode, DL, VT,
                                     {MaskedOff, Src1, Src2, V0, VL, SEW, Glue}),
              0);
          ReplaceNode(Node, CurDAG->getMachineNode(VMXOROpcode, DL, VT,
                                                   {Cmp, Mask, VL, MaskSEW}));
          return;
        }
      }
      break;
    }
    case ISD::INTRINSIC_W_CHAIN: {
      LLVM_DEBUG(dbgs() << "select for intrinsic with chain\n");
      unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
      switch (IntNo) {
        // By default we do not custom select any intrinsic.
      default:
        break;
      #define BFU_WITH_CHAIN
        //    #include "PrimateBFUTypeLegalizer.inc"
      #undef BFU_WITH_CHAIN
      // case Intrinsic::primate_input: {
      //   LLVM_DEBUG(dbgs() << "input lower w/ chain thing\n");
      //   LLVM_DEBUG(dbgs() << "Node num vals: " << Node->getNumValues() << "\n");

      //   auto cnst = CurDAG->getTargetConstant(Node->getConstantOperandVal(1), DL, MVT::i32);

      //   MachineSDNode* inputNode = CurDAG->getMachineNode(Primate::INPUT_READ, DL, Node->getVTList(), cnst);

      //   ReplaceNode(Node, inputNode);

      //   return;
      // }
      // case Intrinsic::primate_BFU_1: {
      //   LLVM_DEBUG(dbgs() << "BFU1 lower w/ chain thing\n");
      //   LLVM_DEBUG(dbgs() << "Node num ops: "  << Node->getNumOperands() << "\n");
      //   LLVM_DEBUG(dbgs() << "Node num vals: " << Node->getNumValues() << "\n");
      //   LLVM_DEBUG(dbgs() << "op 1 value size: " << Node->getOperand(1).getValueType().getSizeInBits() << "\n");
      //   return;
      // }
      // case Intrinsic::primate_BFU_2: {
      //   LLVM_DEBUG(dbgs() << "BFU2 lower w/ chain thing\n");
      //   LLVM_DEBUG(dbgs() << "Node num ops: "  << Node->getNumOperands() << "\n");
      //   LLVM_DEBUG(dbgs() << "Node num vals: " << Node->getNumValues() << "\n");
      //   LLVM_DEBUG(dbgs() << "op 1 value size: " << Node->getOperand(1).getValueType().getSizeInBits() << "\n");
      //   return;
      // }

      case Intrinsic::primate_vsetvli:
      case Intrinsic::primate_vsetvlimax: {
        llvm_unreachable("Primate should not generate VSETVLI{MAX} intrinsics");
      }
      case Intrinsic::primate_vlseg2:
      case Intrinsic::primate_vlseg3:
      case Intrinsic::primate_vlseg4:
      case Intrinsic::primate_vlseg5:
      case Intrinsic::primate_vlseg6:
      case Intrinsic::primate_vlseg7:
      case Intrinsic::primate_vlseg8: {
        selectVLSEG(Node, /*IsMasked*/ false, /*IsStrided*/ false);
        return;
      }
      case Intrinsic::primate_vlseg2_mask:
      case Intrinsic::primate_vlseg3_mask:
      case Intrinsic::primate_vlseg4_mask:
      case Intrinsic::primate_vlseg5_mask:
      case Intrinsic::primate_vlseg6_mask:
      case Intrinsic::primate_vlseg7_mask:
      case Intrinsic::primate_vlseg8_mask: {
        selectVLSEG(Node, /*IsMasked*/ true, /*IsStrided*/ false);
        return;
      }
      case Intrinsic::primate_vlsseg2:
      case Intrinsic::primate_vlsseg3:
      case Intrinsic::primate_vlsseg4:
      case Intrinsic::primate_vlsseg5:
      case Intrinsic::primate_vlsseg6:
      case Intrinsic::primate_vlsseg7:
      case Intrinsic::primate_vlsseg8: {
        selectVLSEG(Node, /*IsMasked*/ false, /*IsStrided*/ true);
        return;
      }
      case Intrinsic::primate_vlsseg2_mask:
      case Intrinsic::primate_vlsseg3_mask:
      case Intrinsic::primate_vlsseg4_mask:
      case Intrinsic::primate_vlsseg5_mask:
      case Intrinsic::primate_vlsseg6_mask:
      case Intrinsic::primate_vlsseg7_mask:
      case Intrinsic::primate_vlsseg8_mask: {
        selectVLSEG(Node, /*IsMasked*/ true, /*IsStrided*/ true);
        return;
      }
      case Intrinsic::primate_vloxseg2:
      case Intrinsic::primate_vloxseg3:
      case Intrinsic::primate_vloxseg4:
      case Intrinsic::primate_vloxseg5:
      case Intrinsic::primate_vloxseg6:
      case Intrinsic::primate_vloxseg7:
      case Intrinsic::primate_vloxseg8:
        selectVLXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ true);
        return;
      case Intrinsic::primate_vluxseg2:
      case Intrinsic::primate_vluxseg3:
      case Intrinsic::primate_vluxseg4:
      case Intrinsic::primate_vluxseg5:
      case Intrinsic::primate_vluxseg6:
      case Intrinsic::primate_vluxseg7:
      case Intrinsic::primate_vluxseg8:
        selectVLXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ false);
        return;
      case Intrinsic::primate_vloxseg2_mask:
      case Intrinsic::primate_vloxseg3_mask:
      case Intrinsic::primate_vloxseg4_mask:
      case Intrinsic::primate_vloxseg5_mask:
      case Intrinsic::primate_vloxseg6_mask:
      case Intrinsic::primate_vloxseg7_mask:
      case Intrinsic::primate_vloxseg8_mask:
        selectVLXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ true);
        return;
      case Intrinsic::primate_vluxseg2_mask:
      case Intrinsic::primate_vluxseg3_mask:
      case Intrinsic::primate_vluxseg4_mask:
      case Intrinsic::primate_vluxseg5_mask:
      case Intrinsic::primate_vluxseg6_mask:
      case Intrinsic::primate_vluxseg7_mask:
      case Intrinsic::primate_vluxseg8_mask:
        selectVLXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ false);
        return;
      case Intrinsic::primate_vlseg8ff:
      case Intrinsic::primate_vlseg7ff:
      case Intrinsic::primate_vlseg6ff:
      case Intrinsic::primate_vlseg5ff:
      case Intrinsic::primate_vlseg4ff:
      case Intrinsic::primate_vlseg3ff:
      case Intrinsic::primate_vlseg2ff: {
        selectVLSEGFF(Node, /*IsMasked*/ false);
        return;
      }
      case Intrinsic::primate_vlseg8ff_mask:
      case Intrinsic::primate_vlseg7ff_mask:
      case Intrinsic::primate_vlseg6ff_mask:
      case Intrinsic::primate_vlseg5ff_mask:
      case Intrinsic::primate_vlseg4ff_mask:
      case Intrinsic::primate_vlseg3ff_mask:
      case Intrinsic::primate_vlseg2ff_mask: {
        selectVLSEGFF(Node, /*IsMasked*/ true);
        return;
      }
      case Intrinsic::primate_vloxei:
      case Intrinsic::primate_vloxei_mask:
      case Intrinsic::primate_vluxei:
      case Intrinsic::primate_vluxei_mask: {
        llvm_unreachable("Primate Should not generate VL{U/O}XEI{_MASK} Intrinsics");
      }
      case Intrinsic::primate_vle1:
      case Intrinsic::primate_vle:
      case Intrinsic::primate_vle_mask:
      case Intrinsic::primate_vlse:
      case Intrinsic::primate_vlse_mask: {
        llvm_unreachable("Primate should not generate VL{S}E_MASK Intrinsics");
      }
      case Intrinsic::primate_vleff:
      case Intrinsic::primate_vleff_mask: {
        llvm_unreachable("Primate should not generate VLEFF{_MASK} Intrincs");
      }
      }
      break;
    }
    case ISD::INTRINSIC_VOID: {
      unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
      switch (IntNo) {
      case Intrinsic::primate_vsseg2:
      case Intrinsic::primate_vsseg3:
      case Intrinsic::primate_vsseg4:
      case Intrinsic::primate_vsseg5:
      case Intrinsic::primate_vsseg6:
      case Intrinsic::primate_vsseg7:
      case Intrinsic::primate_vsseg8: {
        selectVSSEG(Node, /*IsMasked*/ false, /*IsStrided*/ false);
        return;
      }
      case Intrinsic::primate_vsseg2_mask:
      case Intrinsic::primate_vsseg3_mask:
      case Intrinsic::primate_vsseg4_mask:
      case Intrinsic::primate_vsseg5_mask:
      case Intrinsic::primate_vsseg6_mask:
      case Intrinsic::primate_vsseg7_mask:
      case Intrinsic::primate_vsseg8_mask: {
        selectVSSEG(Node, /*IsMasked*/ true, /*IsStrided*/ false);
        return;
      }
      case Intrinsic::primate_vssseg2:
      case Intrinsic::primate_vssseg3:
      case Intrinsic::primate_vssseg4:
      case Intrinsic::primate_vssseg5:
      case Intrinsic::primate_vssseg6:
      case Intrinsic::primate_vssseg7:
      case Intrinsic::primate_vssseg8: {
        selectVSSEG(Node, /*IsMasked*/ false, /*IsStrided*/ true);
        return;
      }
      case Intrinsic::primate_vssseg2_mask:
      case Intrinsic::primate_vssseg3_mask:
      case Intrinsic::primate_vssseg4_mask:
      case Intrinsic::primate_vssseg5_mask:
      case Intrinsic::primate_vssseg6_mask:
      case Intrinsic::primate_vssseg7_mask:
      case Intrinsic::primate_vssseg8_mask: {
        selectVSSEG(Node, /*IsMasked*/ true, /*IsStrided*/ true);
        return;
      }
      case Intrinsic::primate_vsoxseg2:
      case Intrinsic::primate_vsoxseg3:
      case Intrinsic::primate_vsoxseg4:
      case Intrinsic::primate_vsoxseg5:
      case Intrinsic::primate_vsoxseg6:
      case Intrinsic::primate_vsoxseg7:
      case Intrinsic::primate_vsoxseg8:
        selectVSXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ true);
        return;
      case Intrinsic::primate_vsuxseg2:
      case Intrinsic::primate_vsuxseg3:
      case Intrinsic::primate_vsuxseg4:
      case Intrinsic::primate_vsuxseg5:
      case Intrinsic::primate_vsuxseg6:
      case Intrinsic::primate_vsuxseg7:
      case Intrinsic::primate_vsuxseg8:
        selectVSXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ false);
        return;
      case Intrinsic::primate_vsoxseg2_mask:
      case Intrinsic::primate_vsoxseg3_mask:
      case Intrinsic::primate_vsoxseg4_mask:
      case Intrinsic::primate_vsoxseg5_mask:
      case Intrinsic::primate_vsoxseg6_mask:
      case Intrinsic::primate_vsoxseg7_mask:
      case Intrinsic::primate_vsoxseg8_mask:
        selectVSXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ true);
        return;
      case Intrinsic::primate_vsuxseg2_mask:
      case Intrinsic::primate_vsuxseg3_mask:
      case Intrinsic::primate_vsuxseg4_mask:
      case Intrinsic::primate_vsuxseg5_mask:
      case Intrinsic::primate_vsuxseg6_mask:
      case Intrinsic::primate_vsuxseg7_mask:
      case Intrinsic::primate_vsuxseg8_mask:
        selectVSXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ false);
        return;
      case Intrinsic::primate_vsoxei:
      case Intrinsic::primate_vsoxei_mask:
      case Intrinsic::primate_vsuxei:
      case Intrinsic::primate_vsuxei_mask: {
        llvm_unreachable("Primate should not generate VS{O/U}XEI{_MASK} Intrinsics");
        // bool IsMasked = IntNo == Intrinsic::primate_vsoxei_mask ||
        //                 IntNo == Intrinsic::primate_vsuxei_mask;
        // bool IsOrdered = IntNo == Intrinsic::primate_vsoxei ||
        //                  IntNo == Intrinsic::primate_vsoxei_mask;

        // MVT VT = Node->getOperand(2)->getSimpleValueType(0);
        // unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

        // unsigned CurOp = 2;
        // SmallVector<SDValue, 8> Operands;
        // Operands.push_back(Node->getOperand(CurOp++)); // Store value.

        // MVT IndexVT;
        // addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
        //                            /*IsStridedOrIndexed*/ true, Operands,
        //                            &IndexVT);

        // assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
        //        "Element count mismatch");

        // PrimateII::VLMUL LMUL = PrimateTargetLowering::getLMUL(VT);
        // PrimateII::VLMUL IndexLMUL = PrimateTargetLowering::getLMUL(IndexVT);
        // unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
        // const Primate::VLX_VSXPseudo *P = Primate::getVSXPseudo(
        //     IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
        //     static_cast<unsigned>(IndexLMUL));
        // MachineSDNode *Store =
        //     CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

        // if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        //   CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

        // ReplaceNode(Node, Store);
        // return;
      }
      case Intrinsic::primate_vse1:
      case Intrinsic::primate_vse:
      case Intrinsic::primate_vse_mask:
      case Intrinsic::primate_vsse:
      case Intrinsic::primate_vsse_mask: {
        llvm_unreachable("Primate should not generate VS{S}E{_MASK} Intrincs");
        // bool IsMasked = IntNo == Intrinsic::primate_vse_mask ||
        //                 IntNo == Intrinsic::primate_vsse_mask;
        // bool IsStrided =
        //     IntNo == Intrinsic::primate_vsse || IntNo == Intrinsic::primate_vsse_mask;

        // MVT VT = Node->getOperand(2)->getSimpleValueType(0);
        // unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

        // unsigned CurOp = 2;
        // SmallVector<SDValue, 8> Operands;
        // Operands.push_back(Node->getOperand(CurOp++)); // Store value.

        // addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
        //                            Operands);

        // PrimateII::VLMUL LMUL = PrimateTargetLowering::getLMUL(VT);
        // const Primate::VSEPseudo *P = Primate::getVSEPseudo(
        //     IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
        // MachineSDNode *Store =
        //     CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);
        // if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        //   CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

        // ReplaceNode(Node, Store);
        // return;
      }
      }
      break;
    }
    case ISD::BITCAST: {
      MVT SrcVT = Node->getOperand(0).getSimpleValueType();
      // Just drop bitcasts between vectors if both are fixed or both are
      // scalable.
      if ((VT.isScalableVector() && SrcVT.isScalableVector()) ||
          (VT.isFixedLengthVector() && SrcVT.isFixedLengthVector())) {
        ReplaceUses(SDValue(Node, 0), Node->getOperand(0));
        CurDAG->RemoveDeadNode(Node);
        return;
      }
      break;
    }
    case ISD::INSERT_SUBVECTOR: {
      SDValue V = Node->getOperand(0);
      SDValue SubV = Node->getOperand(1);
      SDLoc DL(SubV);
      auto Idx = Node->getConstantOperandVal(2);
      MVT SubVecVT = SubV.getSimpleValueType();

      const PrimateTargetLowering &TLI = *Subtarget->getTargetLowering();
      MVT SubVecContainerVT = SubVecVT;
      // Establish the correct scalable-vector types for any fixed-length type.
      if (SubVecVT.isFixedLengthVector())
        SubVecContainerVT = TLI.getContainerForFixedLengthVector(SubVecVT);
      if (VT.isFixedLengthVector())
        VT = TLI.getContainerForFixedLengthVector(VT);

      const auto *TRI = Subtarget->getRegisterInfo();
      unsigned SubRegIdx;
      std::tie(SubRegIdx, Idx) =
          PrimateTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
              VT, SubVecContainerVT, Idx, TRI);

      // If the Idx hasn't been completely eliminated then this is a subvector
      // insert which doesn't naturally align to a vector register. These must
      // be handled using instructions to manipulate the vector registers.
      if (Idx != 0)
        break;

      PrimateII::VLMUL SubVecLMUL = PrimateTargetLowering::getLMUL(SubVecContainerVT);
      bool IsSubVecPartReg = SubVecLMUL == PrimateII::VLMUL::LMUL_F2 ||
                             SubVecLMUL == PrimateII::VLMUL::LMUL_F4 ||
                             SubVecLMUL == PrimateII::VLMUL::LMUL_F8;
      (void)IsSubVecPartReg; // Silence unused variable warning without asserts.
      assert((!IsSubVecPartReg || V.isUndef()) &&
             "Expecting lowering to have created legal INSERT_SUBVECTORs when "
             "the subvector is smaller than a full-sized register");

      // If we haven't set a SubRegIdx, then we must be going between
      // equally-sized LMUL groups (e.g. VR -> VR). This can be done as a copy.
      if (SubRegIdx == Primate::NoSubRegister) {
        unsigned InRegClassID = PrimateTargetLowering::getRegClassIDForVecVT(VT);
        assert(PrimateTargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                   InRegClassID &&
               "Unexpected subvector extraction");
        SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
        SDNode *NewNode = CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                                 DL, VT, SubV, RC);
        ReplaceNode(Node, NewNode);
        return;
      }

      SDValue Insert = CurDAG->getTargetInsertSubreg(SubRegIdx, DL, VT, V, SubV);
      ReplaceNode(Node, Insert.getNode());
      return;
    }
    case ISD::EXTRACT_SUBVECTOR: {
      SDValue V = Node->getOperand(0);
      auto Idx = Node->getConstantOperandVal(1);
      MVT InVT = V.getSimpleValueType();
      SDLoc DL(V);

      const PrimateTargetLowering &TLI = *Subtarget->getTargetLowering();
      MVT SubVecContainerVT = VT;
      // Establish the correct scalable-vector types for any fixed-length type.
      if (VT.isFixedLengthVector())
        SubVecContainerVT = TLI.getContainerForFixedLengthVector(VT);
      if (InVT.isFixedLengthVector())
        InVT = TLI.getContainerForFixedLengthVector(InVT);

      const auto *TRI = Subtarget->getRegisterInfo();
      unsigned SubRegIdx;
      std::tie(SubRegIdx, Idx) =
          PrimateTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
              InVT, SubVecContainerVT, Idx, TRI);

      // If the Idx hasn't been completely eliminated then this is a subvector
      // extract which doesn't naturally align to a vector register. These must
      // be handled using instructions to manipulate the vector registers.
      if (Idx != 0)
        break;

      // If we haven't set a SubRegIdx, then we must be going between
      // equally-sized LMUL types (e.g. VR -> VR). This can be done as a copy.
      if (SubRegIdx == Primate::NoSubRegister) {
        unsigned InRegClassID = PrimateTargetLowering::getRegClassIDForVecVT(InVT);
        assert(PrimateTargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                   InRegClassID &&
               "Unexpected subvector extraction");
        SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
        SDNode *NewNode =
            CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, DL, VT, V, RC);
        ReplaceNode(Node, NewNode);
        return;
      }

      SDValue Extract = CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, V);
      ReplaceNode(Node, Extract.getNode());
      return;
    }
    // case PrimateISD::VMV_V_X_VL:
    // case PrimateISD::VFMV_V_F_VL: {
    //   // Try to match splat of a scalar load to a strided load with stride of x0.
    //   SDValue Src = Node->getOperand(0);
    //   auto *Ld = dyn_cast<LoadSDNode>(Src);
    //   if (!Ld)
    //     break;
    //   EVT MemVT = Ld->getMemoryVT();
    //   // The memory VT should be the same size as the element type.
    //   if (MemVT.getStoreSize() != VT.getVectorElementType().getStoreSize())
    //     break;
    //   if (!IsProfitableToFold(Src, Node, Node) ||
    //       !IsLegalToFold(Src, Node, Node, TM.getOptLevel()))
    //     break;

    //   SDValue VL;
    //   selectVLOp(Node->getOperand(1), VL);

    //   unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
    //   SDValue SEW = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);

    //   SDValue Operands[] = {Ld->getBasePtr(),
    //                         CurDAG->getRegister(Primate::X0, XLenVT), VL, SEW,
    //                         Ld->getChain()};

    //   PrimateII::VLMUL LMUL = PrimateTargetLowering::getLMUL(VT);
    //   const Primate::VLEPseudo *P = Primate::getVLEPseudo(
    //       /*IsMasked*/ false, /*IsStrided*/ true, /*FF*/ false, Log2SEW,
    //       static_cast<unsigned>(LMUL));
    //   MachineSDNode *Load =
    //       CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

    //   if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    //     CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

    //   ReplaceNode(Node, Load);
    //   return;
    // }
    // case ISD::EXTRACT_VALUE: {
    //   // This is used for things
    //   SmallVector<SDValue> operands;
    //   operands.push_back(Node->getOperand(0));
    //   operands.push_back(curDAG->getConst Node->getConstantOperandVal(1));
    //   MachineSDNode *extNode = CurDAG->getMachineNode(Primate::EXTRACT, DL, Node->getVTList(), operands);
    //   ReplaceNode(Node, extNode);
    //   return;
    // }
    // case ISD::INSERT_VALUE: {
    //   // This is used for things
    //   SmallVector<SDValue> operands;
    //   operands.push_back(Node->getOperand(0));
    //   operands.push_back(Node->getOperand(1));
    //   operands.push_back(Node->getOperand(2));
    //   MachineSDNode *extNode = CurDAG->getMachineNode(Primate::INSERT, DL, Node->getVTList(), operands);
    //   ReplaceNode(Node, extNode);
    //   return;
    // }
  }

  // Select the default instruction.
  LLVM_DEBUG(dbgs() << "missed a custom ISEL hook: ");
  LLVM_DEBUG(Node->dump());
  SelectCode(Node);
}

bool PrimateDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, InlineAsm::ConstraintCode ConstraintID,
    std::vector<SDValue> &OutOps) {
  // Always produce a register and immediate operand, as expected by
  // PrimateAsmPrinter::PrintAsmMemoryOperand.
  switch (ConstraintID) {
  case InlineAsm::ConstraintCode::o:
  case InlineAsm::ConstraintCode::m: {
    SDValue Op0, Op1;
    bool Found = SelectAddrRegImm(Op, Op0, Op1);
    assert(Found && "SelectAddrRegImm should always succeed");
    (void)Found;
    OutOps.push_back(Op0);
    OutOps.push_back(Op1);
    return false;
  }
  case InlineAsm::ConstraintCode::A:
    OutOps.push_back(Op);
    OutOps.push_back(
        CurDAG->getTargetConstant(0, SDLoc(Op), Subtarget->getXLenVT()));
    return false;
  default:
    report_fatal_error("Unexpected asm memory constraint " +
                       InlineAsm::getMemConstraintName(ConstraintID));
  }

  return true;
}

// bool PrimateDAGToDAGISel::SelectInlineAsmMemoryOperand(
//     const SDValue &Op, unsigned ConstraintID, std::vector<SDValue> &OutOps) {
//   switch (ConstraintID) {
//   case InlineAsm::Constraint_m:
//     // We just support simple memory operands that have a single address
//     // operand and need no special handling.
//     OutOps.push_back(Op);
//     return false;
//   case InlineAsm::Constraint_A:
//     OutOps.push_back(Op);
//     return false;
//   default:
//     break;
//   }

//   return true;
// }

static SDValue selectImmSeq(SelectionDAG *CurDAG, const SDLoc &DL, const MVT VT,
                            PrimateMatInt::InstSeq &Seq) {
  SDValue SrcReg = CurDAG->getRegister(Primate::X0, VT);
  for (const PrimateMatInt::Inst &Inst : Seq) {
    SDValue SDImm = CurDAG->getTargetConstant(Inst.getImm(), DL, VT);
    SDNode *Result = nullptr;
    switch (Inst.getOpndKind()) {
    case PrimateMatInt::Imm:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SDImm);
      break;
    case PrimateMatInt::RegX0:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg,
                                      CurDAG->getRegister(Primate::X0, VT));
      break;
    case PrimateMatInt::RegReg:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg, SrcReg);
      break;
    case PrimateMatInt::RegImm:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg, SDImm);
      break;
    }

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return SrcReg;
}

static SDValue selectImm(SelectionDAG *CurDAG, const SDLoc &DL, const MVT VT,
                         int64_t Imm, const PrimateSubtarget &Subtarget) {
  PrimateMatInt::InstSeq Seq = PrimateMatInt::generateInstSeq(Imm, Subtarget);

  // See if we can create this constant as (ADD (SLLI X, C), X) where X is at
  // worst an LUI+ADDIW. This will require an extra register, but avoids a
  // constant pool.
  // If we have Zba we can use (ADD_UW X, (SLLI X, 32)) to handle cases where
  // low and high 32 bits are the same and bit 31 and 63 are set.
  if (Seq.size() > 3) {
    unsigned ShiftAmt, AddOpc;
    PrimateMatInt::InstSeq SeqLo =
        PrimateMatInt::generateTwoRegInstSeq(Imm, Subtarget, ShiftAmt, AddOpc);
    if (!SeqLo.empty() && (SeqLo.size() + 2) < Seq.size()) {
      SDValue Lo = selectImmSeq(CurDAG, DL, VT, SeqLo);

      SDValue SLLI = SDValue(
          CurDAG->getMachineNode(Primate::SLLI, DL, VT, Lo,
                                 CurDAG->getTargetConstant(ShiftAmt, DL, VT)),
          0);
      return SDValue(CurDAG->getMachineNode(AddOpc, DL, VT, Lo, SLLI), 0);
    }
  }

  // Otherwise, use the original sequence.
  return selectImmSeq(CurDAG, DL, VT, Seq);
}

// Fold constant addresses.
static bool selectConstantAddr(SelectionDAG *CurDAG, const SDLoc &DL,
                               const MVT VT, const PrimateSubtarget *Subtarget,
                               SDValue Addr, SDValue &Base, SDValue &Offset,
                               bool IsPrefetch = false) {
  if (!isa<ConstantSDNode>(Addr))
    return false;

  int64_t CVal = cast<ConstantSDNode>(Addr)->getSExtValue();

  // If the constant is a simm12, we can fold the whole constant and use X0 as
  // the base. If the constant can be materialized with LUI+simm12, use LUI as
  // the base. We can't use generateInstSeq because it favors LUI+ADDIW.
  int64_t Lo12 = SignExtend64<12>(CVal);
  int64_t Hi = (uint64_t)CVal - (uint64_t)Lo12;
  if (!Subtarget->is64Bit() || isInt<32>(Hi)) {
    if (IsPrefetch && (Lo12 & 0b11111) != 0)
      return false;

    if (Hi) {
      int64_t Hi20 = (Hi >> 12) & 0xfffff;
      Base = SDValue(
          CurDAG->getMachineNode(Primate::LUI, DL, VT,
                                 CurDAG->getTargetConstant(Hi20, DL, VT)),
          0);
    } else {
      Base = CurDAG->getRegister(Primate::X0, VT);
    }
    Offset = CurDAG->getTargetConstant(Lo12, DL, VT);
    return true;
  }

  // Ask how constant materialization would handle this constant.
  PrimateMatInt::InstSeq Seq = PrimateMatInt::generateInstSeq(CVal, *Subtarget);

  // If the last instruction would be an ADDI, we can fold its immediate and
  // emit the rest of the sequence as the base.
  if (Seq.back().getOpcode() != Primate::ADDI)
    return false;
  Lo12 = Seq.back().getImm();
  if (IsPrefetch && (Lo12 & 0b11111) != 0)
    return false;

  // Drop the last instruction.
  Seq.pop_back();
  assert(!Seq.empty() && "Expected more instructions in sequence");

  Base = selectImmSeq(CurDAG, DL, VT, Seq);
  Offset = CurDAG->getTargetConstant(Lo12, DL, VT);
  return true;
}

// Is this ADD instruction only used as the base pointer of scalar loads and
// stores?
static bool isWorthFoldingAdd(SDValue Add) {
  for (auto *Use : Add->uses()) {
    if (Use->getOpcode() != ISD::LOAD && Use->getOpcode() != ISD::STORE &&
        Use->getOpcode() != ISD::ATOMIC_LOAD &&
        Use->getOpcode() != ISD::ATOMIC_STORE)
      return false;
    EVT VT = cast<MemSDNode>(Use)->getMemoryVT();
    if (!VT.isScalarInteger() && VT != MVT::f16 && VT != MVT::f32 &&
        VT != MVT::f64)
      return false;
    // Don't allow stores of the value. It must be used as the address.
    if (Use->getOpcode() == ISD::STORE &&
        cast<StoreSDNode>(Use)->getValue() == Add)
      return false;
    if (Use->getOpcode() == ISD::ATOMIC_STORE &&
        cast<AtomicSDNode>(Use)->getVal() == Add)
      return false;
  }

  return true;
}

bool PrimateDAGToDAGISel::SelectAddrFI(SDValue Addr, SDValue &Base) {
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    return true;
  }
  return false;
}

bool PrimateDAGToDAGISel::SelectBaseAddr(SDValue Addr, SDValue &Base) {
  // If this is FrameIndex, select it directly. Otherwise just let it get
  // selected to a register independently.
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr))
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
  else
    Base = Addr;
  return true;
}

bool PrimateDAGToDAGISel::selectShiftMask(SDValue N, unsigned ShiftWidth,
                                        SDValue &ShAmt) {
  // Shift instructions on Primate only read the lower 5 or 6 bits of the shift
  // amount. If there is an AND on the shift amount, we can bypass it if it
  // doesn't affect any of those bits.
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1))) {
    const APInt &AndMask = N->getConstantOperandAPInt(1);

    // Since the max shift amount is a power of 2 we can subtract 1 to make a
    // mask that covers the bits needed to represent all shift amounts.
    assert(isPowerOf2_32(ShiftWidth) && "Unexpected max shift amount!");
    APInt ShMask(AndMask.getBitWidth(), ShiftWidth - 1);

    if (ShMask.isSubsetOf(AndMask)) {
      ShAmt = N.getOperand(0);
      return true;
    }

    // SimplifyDemandedBits may have optimized the mask so try restoring any
    // bits that are known zero.
    KnownBits Known = CurDAG->computeKnownBits(N->getOperand(0));
    if (ShMask.isSubsetOf(AndMask | Known.Zero)) {
      ShAmt = N.getOperand(0);
      return true;
    }
  }

  ShAmt = N;
  return true;
}

bool PrimateDAGToDAGISel::SelectAddrFrameIndex(SDValue Addr, SDValue &Base,
                                             SDValue &Offset) {
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), Subtarget->getXLenVT());
    return true;
  }

  return false;
}

bool PrimateDAGToDAGISel::SelectAddrRegImm(SDValue Addr, SDValue &Base,
                                         SDValue &Offset, bool IsINX) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  SDLoc DL(Addr);
  MVT VT = Addr.getSimpleValueType();

  if (Addr.getOpcode() == PrimateISD::ADD_LO) {
    Base = Addr.getOperand(0);
    Offset = Addr.getOperand(1);
    return true;
  }

  int64_t RV32ZdinxRange = IsINX ? 4 : 0;
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal) && isInt<12>(CVal + RV32ZdinxRange)) {
      Base = Addr.getOperand(0);
      if (Base.getOpcode() == PrimateISD::ADD_LO) {
        SDValue LoOperand = Base.getOperand(1);
        if (auto *GA = dyn_cast<GlobalAddressSDNode>(LoOperand)) {
          // If the Lo in (ADD_LO hi, lo) is a global variable's address
          // (its low part, really), then we can rely on the alignment of that
          // variable to provide a margin of safety before low part can overflow
          // the 12 bits of the load/store offset. Check if CVal falls within
          // that margin; if so (low part + CVal) can't overflow.
          const DataLayout &DL = CurDAG->getDataLayout();
          Align Alignment = commonAlignment(
              GA->getGlobal()->getPointerAlignment(DL), GA->getOffset());
          if (CVal == 0 || Alignment > CVal) {
            int64_t CombinedOffset = CVal + GA->getOffset();
            Base = Base.getOperand(0);
            Offset = CurDAG->getTargetGlobalAddress(
                GA->getGlobal(), SDLoc(LoOperand), LoOperand.getValueType(),
                CombinedOffset, GA->getTargetFlags());
            return true;
          }
        }
      }

      if (auto *FIN = dyn_cast<FrameIndexSDNode>(Base))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      Offset = CurDAG->getTargetConstant(CVal, DL, VT);
      return true;
    }
  }

  // Handle ADD with large immediates.
  if (Addr.getOpcode() == ISD::ADD && isa<ConstantSDNode>(Addr.getOperand(1))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    assert(!(isInt<12>(CVal) && isInt<12>(CVal + RV32ZdinxRange)) &&
           "simm12 not already handled?");

    // Handle immediates in the range [-4096,-2049] or [2048, 4094]. We can use
    // an ADDI for part of the offset and fold the rest into the load/store.
    // This mirrors the AddiPair PatFrag in PrimateInstrInfo.td.
    if (isInt<12>(CVal / 2) && isInt<12>(CVal - CVal / 2)) {
      int64_t Adj = CVal < 0 ? -2048 : 2047;
      Base = SDValue(
          CurDAG->getMachineNode(Primate::ADDI, DL, VT, Addr.getOperand(0),
                                 CurDAG->getTargetConstant(Adj, DL, VT)),
          0);
      Offset = CurDAG->getTargetConstant(CVal - Adj, DL, VT);
      return true;
    }

    // For larger immediates, we might be able to save one instruction from
    // constant materialization by folding the Lo12 bits of the immediate into
    // the address. We should only do this if the ADD is only used by loads and
    // stores that can fold the lo12 bits. Otherwise, the ADD will get iseled
    // separately with the full materialized immediate creating extra
    // instructions.
    if (isWorthFoldingAdd(Addr) &&
        selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr.getOperand(1), Base,
                           Offset)) {
      // Insert an ADD instruction with the materialized Hi52 bits.
      Base = SDValue(
          CurDAG->getMachineNode(Primate::ADD, DL, VT, Addr.getOperand(0), Base),
          0);
      return true;
    }
  }

  if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr, Base, Offset))
    return true;

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, VT);
  return true;
}

/// Similar to SelectAddrRegImm, except that the least significant 5 bits of
/// Offset shoule be all zeros.
bool PrimateDAGToDAGISel::SelectAddrRegImmLsb00000(SDValue Addr, SDValue &Base,
                                                 SDValue &Offset) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  SDLoc DL(Addr);
  MVT VT = Addr.getSimpleValueType();

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal)) {
      Base = Addr.getOperand(0);

      // Early-out if not a valid offset.
      if ((CVal & 0b11111) != 0) {
        Base = Addr;
        Offset = CurDAG->getTargetConstant(0, DL, VT);
        return true;
      }

      if (auto *FIN = dyn_cast<FrameIndexSDNode>(Base))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      Offset = CurDAG->getTargetConstant(CVal, DL, VT);
      return true;
    }
  }

  // Handle ADD with large immediates.
  if (Addr.getOpcode() == ISD::ADD && isa<ConstantSDNode>(Addr.getOperand(1))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    assert(!(isInt<12>(CVal) && isInt<12>(CVal)) &&
           "simm12 not already handled?");

    // Handle immediates in the range [-4096,-2049] or [2017, 4065]. We can save
    // one instruction by folding adjustment (-2048 or 2016) into the address.
    if ((-2049 >= CVal && CVal >= -4096) || (4065 >= CVal && CVal >= 2017)) {
      int64_t Adj = CVal < 0 ? -2048 : 2016;
      int64_t AdjustedOffset = CVal - Adj;
      Base = SDValue(CurDAG->getMachineNode(
                         Primate::ADDI, DL, VT, Addr.getOperand(0),
                         CurDAG->getTargetConstant(AdjustedOffset, DL, VT)),
                     0);
      Offset = CurDAG->getTargetConstant(Adj, DL, VT);
      return true;
    }

    if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr.getOperand(1), Base,
                           Offset, true)) {
      // Insert an ADD instruction with the materialized Hi52 bits.
      Base = SDValue(
          CurDAG->getMachineNode(Primate::ADD, DL, VT, Addr.getOperand(0), Base),
          0);
      return true;
    }
  }

  if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr, Base, Offset, true))
    return true;

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, VT);
  return true;
}


bool PrimateDAGToDAGISel::selectSExti32(SDValue N, SDValue &Val) {
  if (N.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      cast<VTSDNode>(N.getOperand(1))->getVT() == MVT::i32) {
    Val = N.getOperand(0);
    return true;
  }
  MVT VT = N.getSimpleValueType();
  if (CurDAG->ComputeNumSignBits(N) > (VT.getSizeInBits() - 32)) {
    Val = N;
    return true;
  }

  return false;
}

bool PrimateDAGToDAGISel::selectZExti32(SDValue N, SDValue &Val) {
  if (N.getOpcode() == ISD::AND) {
    auto *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (C && C->getZExtValue() == UINT64_C(0xFFFFFFFF)) {
      Val = N.getOperand(0);
      return true;
    }
  }
  MVT VT = N.getSimpleValueType();
  APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(), 32);
  if (CurDAG->MaskedValueIsZero(N, Mask)) {
    Val = N;
    return true;
  }

  return false;
}

// Select VL as a 5 bit immediate or a value that will become a register. This
// allows us to choose betwen VSETIVLI or VSETVLI later.
bool PrimateDAGToDAGISel::selectVLOp(SDValue N, SDValue &VL) {
  auto *C = dyn_cast<ConstantSDNode>(N);
  if (C && isUInt<5>(C->getZExtValue()))
    VL = CurDAG->getTargetConstant(C->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
  else
    VL = N;

  return true;
}

bool PrimateDAGToDAGISel::selectVSplat(SDValue N, SDValue &SplatVal) {
  llvm_unreachable("Primate should not select a VSplat");
  // if (N.getOpcode() != ISD::SPLAT_VECTOR &&
  //     N.getOpcode() != PrimateISD::SPLAT_VECTOR_I64 &&
  //     N.getOpcode() != PrimateISD::VMV_V_X_VL)
  //   return false;
  // SplatVal = N.getOperand(0);
  // return true;
}

using ValidateFn = bool (*)(int64_t);

static bool selectVSplatSimmHelper(SDValue N, SDValue &SplatVal,
                                   SelectionDAG &DAG,
                                   const PrimateSubtarget &Subtarget,
                                   ValidateFn ValidateImm) {
  llvm_unreachable("Primate should not select a VSplatSimmHelper");
  // if ((N.getOpcode() != ISD::SPLAT_VECTOR &&
  //      N.getOpcode() != PrimateISD::SPLAT_VECTOR_I64 &&
  //      N.getOpcode() != PrimateISD::VMV_V_X_VL) ||
  //     !isa<ConstantSDNode>(N.getOperand(0)))
  //   return false;

  // int64_t SplatImm = cast<ConstantSDNode>(N.getOperand(0))->getSExtValue();

  // // ISD::SPLAT_VECTOR, PrimateISD::SPLAT_VECTOR_I64 and PrimateISD::VMV_V_X_VL
  // // share semantics when the operand type is wider than the resulting vector
  // // element type: an implicit truncation first takes place. Therefore, perform
  // // a manual truncation/sign-extension in order to ignore any truncated bits
  // // and catch any zero-extended immediate.
  // // For example, we wish to match (i8 -1) -> (XLenVT 255) as a simm5 by first
  // // sign-extending to (XLenVT -1).
  // MVT XLenVT = Subtarget.getXLenVT();
  // assert(XLenVT == N.getOperand(0).getSimpleValueType() &&
  //        "Unexpected splat operand type");
  // MVT EltVT = N.getSimpleValueType().getVectorElementType();
  // if (EltVT.bitsLT(XLenVT))
  //   SplatImm = SignExtend64(SplatImm, EltVT.getSizeInBits());

  // if (!ValidateImm(SplatImm))
  //   return false;

  // SplatVal = DAG.getTargetConstant(SplatImm, SDLoc(N), XLenVT);
  // return true;
}

bool PrimateDAGToDAGISel::selectVSplatSimm5(SDValue N, SDValue &SplatVal) {
  return selectVSplatSimmHelper(N, SplatVal, *CurDAG, *Subtarget,
                                [](int64_t Imm) { return isInt<5>(Imm); });
}

bool PrimateDAGToDAGISel::selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal) {
  return selectVSplatSimmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [](int64_t Imm) { return (isInt<5>(Imm) && Imm != -16) || Imm == 16; });
}

bool PrimateDAGToDAGISel::selectVSplatSimm5Plus1NonZero(SDValue N,
                                                      SDValue &SplatVal) {
  return selectVSplatSimmHelper(
      N, SplatVal, *CurDAG, *Subtarget, [](int64_t Imm) {
        return Imm != 0 && ((isInt<5>(Imm) && Imm != -16) || Imm == 16);
      });
}

bool PrimateDAGToDAGISel::selectVSplatUimm5(SDValue N, SDValue &SplatVal) {
  llvm_unreachable("Primate should not select a VSplatUimm5");
  // if ((N.getOpcode() != ISD::SPLAT_VECTOR &&
  //      N.getOpcode() != PrimateISD::SPLAT_VECTOR_I64 &&
  //      N.getOpcode() != PrimateISD::VMV_V_X_VL) ||
  //     !isa<ConstantSDNode>(N.getOperand(0)))
  //   return false;

  // int64_t SplatImm = cast<ConstantSDNode>(N.getOperand(0))->getSExtValue();

  // if (!isUInt<5>(SplatImm))
  //   return false;

  // SplatVal =
  //     CurDAG->getTargetConstant(SplatImm, SDLoc(N), Subtarget->getXLenVT());

  // return true;
}

bool PrimateDAGToDAGISel::selectPRVSimm5(SDValue N, unsigned Width,
                                       SDValue &Imm) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    int64_t ImmVal = SignExtend64(C->getSExtValue(), Width);

    if (!isInt<5>(ImmVal))
      return false;

    Imm = CurDAG->getTargetConstant(ImmVal, SDLoc(N), Subtarget->getXLenVT());
    return true;
  }

  return false;
}

// Merge an ADDI into the offset of a load/store instruction where possible.
// (load (addi base, off1), off2) -> (load base, off1+off2)
// (store val, (addi base, off1), off2) -> (store val, base, off1+off2)
// This is possible when off1+off2 fits a 12-bit immediate.
void PrimateDAGToDAGISel::doPeepholeLoadStoreADDI() {
  SelectionDAG::allnodes_iterator Position(CurDAG->getRoot().getNode());
  ++Position;

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    int OffsetOpIdx;
    int BaseOpIdx;

    // Only attempt this optimisation for I-type loads and S-type stores.
    switch (N->getMachineOpcode()) {
    default:
      continue;
    case Primate::LB:
    case Primate::LH:
    case Primate::LW:
    case Primate::LBU:
    case Primate::LHU:
    case Primate::LWU:
    case Primate::LD:
    case Primate::FLH:
    case Primate::FLW:
    case Primate::FLD:
      BaseOpIdx = 0;
      OffsetOpIdx = 1;
      break;
    case Primate::SB:
    case Primate::SH:
    case Primate::SW:
    case Primate::SD:
    case Primate::FSH:
    case Primate::FSW:
    case Primate::FSD:
      BaseOpIdx = 1;
      OffsetOpIdx = 2;
      break;
    }

    if (!isa<ConstantSDNode>(N->getOperand(OffsetOpIdx)))
      continue;

    SDValue Base = N->getOperand(BaseOpIdx);

    // If the base is an ADDI, we can merge it in to the load/store.
    if (!Base.isMachineOpcode() || Base.getMachineOpcode() != Primate::ADDI)
      continue;

    SDValue ImmOperand = Base.getOperand(1);
    uint64_t Offset2 = N->getConstantOperandVal(OffsetOpIdx);

    if (auto *Const = dyn_cast<ConstantSDNode>(ImmOperand)) {
      int64_t Offset1 = Const->getSExtValue();
      int64_t CombinedOffset = Offset1 + Offset2;
      if (!isInt<12>(CombinedOffset))
        continue;
      ImmOperand = CurDAG->getTargetConstant(CombinedOffset, SDLoc(ImmOperand),
                                             ImmOperand.getValueType());
    } else if (auto *GA = dyn_cast<GlobalAddressSDNode>(ImmOperand)) {
      // If the off1 in (addi base, off1) is a global variable's address (its
      // low part, really), then we can rely on the alignment of that variable
      // to provide a margin of safety before off1 can overflow the 12 bits.
      // Check if off2 falls within that margin; if so off1+off2 can't overflow.
      const DataLayout &DL = CurDAG->getDataLayout();
      Align Alignment = GA->getGlobal()->getPointerAlignment(DL);
      if (Offset2 != 0 && Alignment <= Offset2)
        continue;
      int64_t Offset1 = GA->getOffset();
      int64_t CombinedOffset = Offset1 + Offset2;
      ImmOperand = CurDAG->getTargetGlobalAddress(
          GA->getGlobal(), SDLoc(ImmOperand), ImmOperand.getValueType(),
          CombinedOffset, GA->getTargetFlags());
    } else if (auto *CP = dyn_cast<ConstantPoolSDNode>(ImmOperand)) {
      // Ditto.
      Align Alignment = CP->getAlign();
      if (Offset2 != 0 && Alignment <= Offset2)
        continue;
      int64_t Offset1 = CP->getOffset();
      int64_t CombinedOffset = Offset1 + Offset2;
      ImmOperand = CurDAG->getTargetConstantPool(
          CP->getConstVal(), ImmOperand.getValueType(), CP->getAlign(),
          CombinedOffset, CP->getTargetFlags());
    } else {
      continue;
    }

    LLVM_DEBUG(dbgs() << "Folding add-immediate into mem-op:\nBase:    ");
    LLVM_DEBUG(Base->dump(CurDAG));
    LLVM_DEBUG(dbgs() << "\nN: ");
    LLVM_DEBUG(N->dump(CurDAG));
    LLVM_DEBUG(dbgs() << "\n");

    // Modify the offset operand of the load/store.
    if (BaseOpIdx == 0) // Load
      CurDAG->UpdateNodeOperands(N, Base.getOperand(0), ImmOperand,
                                 N->getOperand(2));
    else // Store
      CurDAG->UpdateNodeOperands(N, N->getOperand(0), Base.getOperand(0),
                                 ImmOperand, N->getOperand(3));

    // The add-immediate may now be dead, in which case remove it.
    if (Base.getNode()->use_empty())
      CurDAG->RemoveDeadNode(Base.getNode());
  }
}

// This pass converts a legalized DAG into a Primate-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createPrimateISelDag(PrimateTargetMachine &TM,
                                         CodeGenOptLevel OptLevel) {
  return new PrimateDAGToDAGISel(TM, OptLevel);
}

char PrimateDAGToDAGISel::ID = 0;

INITIALIZE_PASS(PrimateDAGToDAGISel, DEBUG_TYPE, PASS_NAME, false, false)
