//===-- PrimateTargetTransformInfo.cpp - Primate specific TTI -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PrimateTargetTransformInfo.h"
#include "MCTargetDesc/PrimateMatInt.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "primatetti"

InstructionCost PrimateTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                                            TTI::TargetCostKind CostKind) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Otherwise, we check how many instructions it will take to materialise.
  const DataLayout &DL = getDataLayout();
  return PrimateMatInt::getIntMatCost(Imm, DL.getTypeSizeInBits(Ty),
                                    *(getST()));
}

InstructionCost PrimateTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                                const APInt &Imm, Type *Ty,
                                                TTI::TargetCostKind CostKind,
                                                Instruction *Inst) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Some instructions in Primate can take a 12-bit immediate. Some of these are
  // commutative, in others the immediate comes from a specific argument index.
  bool Takes12BitImm = false;
  unsigned ImmArgIdx = ~0U;

  switch (Opcode) {
  case Instruction::GetElementPtr:
    // Never hoist any arguments to a GetElementPtr. CodeGenPrepare will
    // split up large offsets in GEP into better parts than ConstantHoisting
    // can.
    return TTI::TCC_Free;
  case Instruction::Add:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Mul:
    Takes12BitImm = true;
    break;
  case Instruction::Sub:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    Takes12BitImm = true;
    ImmArgIdx = 1;
    break;
  default:
    break;
  }

  if (Takes12BitImm) {
    // Check immediate is the correct argument...
    if (Instruction::isCommutative(Opcode) || Idx == ImmArgIdx) {
      // ... and fits into the 12-bit immediate.
      if (Imm.getSignificantBits() <= 64 &&
          getTLI()->isLegalAddImmediate(Imm.getSExtValue())) {
        return TTI::TCC_Free;
      }
    }

    // Otherwise, use the full materialisation cost.
    return getIntImmCost(Imm, Ty, CostKind);
  }

  // By default, prevent hoisting.
  return TTI::TCC_Free;
}

InstructionCost
PrimateTTIImpl::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                  const APInt &Imm, Type *Ty,
                                  TTI::TargetCostKind CostKind) {
  // Prevent hoisting in unknown cases.
  return TTI::TCC_Free;
}

TargetTransformInfo::PopcntSupportKind
PrimateTTIImpl::getPopcntSupport(unsigned TyWidth) {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return ST->hasStdExtZbb() ? TTI::PSK_FastHardware : TTI::PSK_Software;
}

bool PrimateTTIImpl::shouldExpandReduction(const IntrinsicInst *II) const {
  // Currently, the ExpandReductions pass can't expand scalable-vector
  // reductions, but we still request expansion as PRV doesn't support certain
  // reductions and the SelectionDAG can't legalize them either.
  switch (II->getIntrinsicID()) {
  default:
    return false;
  // These reductions have no equivalent in PRV
  case Intrinsic::vector_reduce_mul:
  case Intrinsic::vector_reduce_fmul:
    return true;
  }
}

std::optional<unsigned> PrimateTTIImpl::getMaxVScale() const {
  // There is no assumption of the maximum vector length in V specification.
  // We use the value specified by users as the maximum vector length.
  // This function will use the assumed maximum vector length to get the
  // maximum vscale for LoopVectorizer.
  // If users do not specify the maximum vector length, we have no way to
  // know whether the LoopVectorizer is safe to do or not.
  // We only consider to use single vector register (LMUL = 1) to vectorize.
  unsigned MaxVectorSizeInBits = ST->getMaxPRVVectorSizeInBits();
  if (ST->hasStdExtV() && MaxVectorSizeInBits != 0)
    return MaxVectorSizeInBits / Primate::PRVBitsPerBlock;
  return BaseT::getMaxVScale();
}

InstructionCost PrimateTTIImpl::getGatherScatterOpCost(
    unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
    Align Alignment, TTI::TargetCostKind CostKind, const Instruction *I) {
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  if ((Opcode == Instruction::Load &&
       !isLegalMaskedGather(DataTy, Align(Alignment))) ||
      (Opcode == Instruction::Store &&
       !isLegalMaskedScatter(DataTy, Align(Alignment))))
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  // FIXME: Only supporting fixed vectors for now.
  if (!isa<FixedVectorType>(DataTy))
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  auto *VTy = cast<FixedVectorType>(DataTy);
  unsigned NumLoads = VTy->getNumElements();
  InstructionCost MemOpCost =
      getMemoryOpCost(Opcode, VTy->getElementType(), Alignment, 0, CostKind, {TTI::OK_AnyValue, TTI::OP_None}, I);
  return NumLoads * MemOpCost;
}
