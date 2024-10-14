//===---- PrimateISelDAGToDAG.h - A dag to dag inst selector for Primate ------===//
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

#ifndef LLVM_LIB_TARGET_PRIMATE_PRIMATEISELDAGTODAG_H
#define LLVM_LIB_TARGET_PRIMATE_PRIMATEISELDAGTODAG_H

#include "Primate.h"
#include "PrimateTargetMachine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Pass.h"
#include "llvm/Support/KnownBits.h"

// Primate-specific code to select Primate machine instructions for
// SelectionDAG operations.
namespace llvm {
class PrimateDAGToDAGISel : public SelectionDAGISel {
  const PrimateSubtarget *Subtarget = nullptr;

  typedef struct {
    SDNode *Pred, *Curr;
    ArrayRef<SDUse> Succ;
  } SubGraphNode_t;

  typedef DenseMap<SDNode *, SubGraphNode_t*> PrimateSubGraph_t;

  SmallVector<PrimateSubGraph_t> SubGraphs;


  SubGraphNode_t *getSubgraphNode(SDNode *CurrNode, SDNode *PrevNode) {
    
  }

  PrimateSubGraph_t *getSubgraph(SDNode *RootNode, SDLoc &DL) {
    PrimateSubGraph_t *SubGraph = new PrimateSubGraph_t;
    SDNode *PrevNode = nullptr;
    SDNode *CurrNode = RootNode;
    // SDNode *NextNode = nullptr;

    // if(CurrNode->use_empty())
    //   return nullptr;
    
    if(SDValue(CurrNode, 0).getValueType() == MVT::i32) {
      dbgs() << "Current Node: ";
      CurrNode->dump();
      dbgs() << "Operation " << CurrNode->getOperationName() 
             << " has "      << CurrNode->getNumOperands() << " OPs:\n";

      for (const auto &OP : CurrNode->ops()) {
        OP.getNode()->dump();
      }
      
      SubGraphNode_t *NewNode = new SubGraphNode_t;
      NewNode->Pred = PrevNode;
      NewNode->Curr = CurrNode;
      NewNode->Succ = CurrNode->ops();
      
      SubGraph->insert({CurrNode, NewNode});
    }
    dbgs() << "\n";
    return nullptr;
  }

public:
  static char ID;

  explicit PrimateDAGToDAGISel(PrimateTargetMachine &TargetMachine, CodeGenOptLevel OptLevel)
      : SelectionDAGISel(ID, TargetMachine, OptLevel) {}

  StringRef getPassName() const override {
    return "Primate DAG->DAG Pattern Instruction Selection";
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<PrimateSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  void PreprocessISelDAG() override;
  void PostprocessISelDAG() override;

  void Select(SDNode *Node) override;

  bool SelectInlineAsmMemoryOperand(const SDValue &Op, InlineAsm::ConstraintCode ConstraintID,
                                    std::vector<SDValue> &OutOps) override;
  
  bool SelectAddrFrameIndex(SDValue Addr, SDValue &Base,
                                             SDValue &Offset);
  bool SelectAddrFI(SDValue Addr, SDValue &Base);
  bool SelectBaseAddr(SDValue Addr, SDValue &Base);
  bool SelectAddrRegImm(SDValue Addr, SDValue &Base, SDValue &Offset,
                        bool IsINX = false);
  bool SelectAddrRegImmINX(SDValue Addr, SDValue &Base, SDValue &Offset) {
    return SelectAddrRegImm(Addr, Base, Offset, true);
  }
  bool SelectAddrRegImmLsb00000(SDValue Addr, SDValue &Base, SDValue &Offset);

  bool selectSHXADD_UWOp(SDValue N, unsigned ShAmt, SDValue &Val);
  template <unsigned ShAmt> bool selectSHXADD_UWOp(SDValue N, SDValue &Val) {
    return selectSHXADD_UWOp(N, ShAmt, Val);
  }

  bool selectShiftMask(SDValue N, unsigned ShiftWidth, SDValue &ShAmt);
  bool selectShiftMaskXLen(SDValue N, SDValue &ShAmt) {
    return selectShiftMask(N, Subtarget->getXLen(), ShAmt);
  }
  bool selectShiftMask32(SDValue N, SDValue &ShAmt) {
    return selectShiftMask(N, 32, ShAmt);
  }

  bool selectSExti32(SDValue N, SDValue &Val);
  bool selectZExti32(SDValue N, SDValue &Val);

  bool selectVLOp(SDValue N, SDValue &VL);

  bool selectVSplat(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5(SDValue N, SDValue &SplatVal);
  bool selectVSplatUimm5(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5Plus1NonZero(SDValue N, SDValue &SplatVal);

  bool selectPRVSimm5(SDValue N, unsigned Width, SDValue &Imm);
  template <unsigned Width> bool selectPRVSimm5(SDValue N, SDValue &Imm) {
    return selectPRVSimm5(N, Width, Imm);
  }

  void addVectorLoadStoreOperands(SDNode *Node, unsigned SEWImm,
                                  const SDLoc &DL, unsigned CurOp,
                                  bool IsMasked, bool IsStridedOrIndexed,
                                  SmallVectorImpl<SDValue> &Operands,
                                  MVT *IndexVT = nullptr);

  void selectVLSEG(SDNode *Node, bool IsMasked, bool IsStrided);
  void selectVLSEGFF(SDNode *Node, bool IsMasked);
  void selectVLXSEG(SDNode *Node, bool IsMasked, bool IsOrdered);
  void selectVSSEG(SDNode *Node, bool IsMasked, bool IsStrided);
  void selectVSXSEG(SDNode *Node, bool IsMasked, bool IsOrdered);

  // Return the RISC-V condition code that matches the given DAG integer
  // condition code. The CondCode must be one of those supported by the RISC-V
  // ISA (see translateSetCCForBranch).
  static PrimateCC::CondCode getPrimateCCForIntCC(ISD::CondCode CC) {
    switch (CC) {
    default:
      llvm_unreachable("Unsupported CondCode");
    case ISD::SETEQ:
      return PrimateCC::COND_EQ;
    case ISD::SETNE:
      return PrimateCC::COND_NE;
    case ISD::SETLT:
      return PrimateCC::COND_LT;
    case ISD::SETGE:
      return PrimateCC::COND_GE;
    case ISD::SETULT:
      return PrimateCC::COND_LTU;
    case ISD::SETUGE:
      return PrimateCC::COND_GEU;
    }
  }

// Include the pieces autogenerated from the target description.
#include "PrimateGenDAGISel.inc"

private:
  void doPeepholeLoadStoreADDI();
};

namespace Primate {
struct VLSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t FF : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VLXSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct VSSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VSXSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct VLEPseudo {
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t FF : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VSEPseudo {
  uint16_t Masked :1;
  uint16_t Strided : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VLX_VSXPseudo {
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

#define GET_PrimateVSSEGTable_DECL
#define GET_PrimateVLSEGTable_DECL
#define GET_PrimateVLXSEGTable_DECL
#define GET_PrimateVSXSEGTable_DECL
#define GET_PrimateVLETable_DECL
#define GET_PrimateVSETable_DECL
#define GET_PrimateVLXTable_DECL
#define GET_PrimateVSXTable_DECL
#include "PrimateGenSearchableTables.inc"
} // namespace Primate

} // namespace llvm

#endif
