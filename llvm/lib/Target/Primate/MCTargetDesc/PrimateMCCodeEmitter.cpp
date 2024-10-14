//===-- PrimateMCCodeEmitter.cpp - Convert Primate code to machine code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PrimateMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PrimateBaseInfo.h"
#include "MCTargetDesc/PrimateFixupKinds.h"
#include "MCTargetDesc/PrimateMCExpr.h"
#include "MCTargetDesc/PrimateMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");
STATISTIC(MCNumFixups, "Number of MC fixups created");

namespace {
class PrimateMCCodeEmitter : public MCCodeEmitter {
  PrimateMCCodeEmitter(const PrimateMCCodeEmitter &) = delete;
  void operator=(const PrimateMCCodeEmitter &) = delete;
  MCContext &Ctx;
  MCInstrInfo const &MCII;

public:
  PrimateMCCodeEmitter(MCContext &ctx, MCInstrInfo const &MCII)
      : Ctx(ctx), MCII(MCII) {}

  ~PrimateMCCodeEmitter() override {}

  virtual void encodeInstruction(const MCInst &Inst, SmallVectorImpl<char> &CB,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const override;

  void expandFunctionCall(const MCInst &MI, SmallVectorImpl<char> &CB,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;

  void expandAddTPRel(const MCInst &MI, SmallVectorImpl<char> &CB,
                      SmallVectorImpl<MCFixup> &Fixups,
                      const MCSubtargetInfo &STI) const;

  /// TableGen'erated function for getting the binary encoding for an
  /// instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  /// Return binary encoding of operand. If the machine operand requires
  /// relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  unsigned getImmOpValueAsr1(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  unsigned getImmOpValue(const MCInst &MI, unsigned OpNo,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;

  unsigned getVMaskReg(const MCInst &MI, unsigned OpNo,
                       SmallVectorImpl<MCFixup> &Fixups,
                       const MCSubtargetInfo &STI) const;

private:
};
} // end anonymous namespace

MCCodeEmitter *llvm::createPrimateMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx) {
  return new PrimateMCCodeEmitter(Ctx, MCII);
}

// Expand PseudoCALL(Reg), PseudoTAIL and PseudoJump to AUIPC and JALR with
// relocation types. We expand those pseudo-instructions while encoding them,
// meaning AUIPC and JALR won't go through Primate MC to MC compressed
// instruction transformation. This is acceptable because AUIPC has no 16-bit
// form and C_JALR has no immediate operand field.  We let linker relaxation
// deal with it. When linker relaxation is enabled, AUIPC and JALR have a
// chance to relax to JAL.
// If the C extension is enabled, JAL has a chance relax to C_JAL.
void PrimateMCCodeEmitter::expandFunctionCall(const MCInst &MI, SmallVectorImpl<char> &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  MCInst TmpInst;
  MCOperand Func;
  MCRegister Ra;
  if (MI.getOpcode() == Primate::PseudoTAIL) {
    Func = MI.getOperand(0);
    Ra = Primate::X6;
  } else if (MI.getOpcode() == Primate::PseudoCALLReg) {
    Func = MI.getOperand(1);
    Ra = MI.getOperand(0).getReg();
  } else if (MI.getOpcode() == Primate::PseudoCALL) {
    Func = MI.getOperand(0);
    Ra = Primate::X1;
  } else if (MI.getOpcode() == Primate::PseudoJump) {
    Func = MI.getOperand(1);
    Ra = MI.getOperand(0).getReg();
  }
  uint32_t Binary;

  assert(Func.isExpr() && "Expected expression");

  const MCExpr *CallExpr = Func.getExpr();

  // Emit AUIPC Ra, Func with R_Primate_CALL relocation type.
  TmpInst = MCInstBuilder(Primate::AUIPC)
                .addReg(Ra)
                .addOperand(MCOperand::createExpr(CallExpr));
  Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);

  if (MI.getOpcode() == Primate::PseudoTAIL ||
      MI.getOpcode() == Primate::PseudoJump)
    // Emit JALR X0, Ra, 0
    TmpInst = MCInstBuilder(Primate::JALR).addReg(Primate::X0).addReg(Ra).addImm(0);
  else
    // Emit JALR Ra, Ra, 0
    TmpInst = MCInstBuilder(Primate::JALR).addReg(Ra).addReg(Ra).addImm(0);
  Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);
}

// Expand PseudoAddTPRel to a simple ADD with the correct relocation.
void PrimateMCCodeEmitter::expandAddTPRel(const MCInst &MI, SmallVectorImpl<char> &CB,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  MCOperand DestReg = MI.getOperand(0);
  MCOperand SrcReg = MI.getOperand(1);
  MCOperand TPReg = MI.getOperand(2);
  assert(TPReg.isReg() && TPReg.getReg() == Primate::X4 &&
         "Expected thread pointer as second input to TP-relative add");

  MCOperand SrcSymbol = MI.getOperand(3);
  assert(SrcSymbol.isExpr() &&
         "Expected expression as third input to TP-relative add");

  const PrimateMCExpr *Expr = dyn_cast<PrimateMCExpr>(SrcSymbol.getExpr());
  assert(Expr && Expr->getKind() == PrimateMCExpr::VK_Primate_TPREL_ADD &&
         "Expected tprel_add relocation on TP-relative symbol");

  // Emit the correct tprel_add relocation for the symbol.
  Fixups.push_back(MCFixup::create(
      0, Expr, MCFixupKind(Primate::fixup_primate_tprel_add), MI.getLoc()));

  // Emit fixup_primate_relax for tprel_add where the relax feature is enabled.
  if (STI.getFeatureBits()[Primate::FeatureRelax]) {
    const MCConstantExpr *Dummy = MCConstantExpr::create(0, Ctx);
    Fixups.push_back(MCFixup::create(
        0, Dummy, MCFixupKind(Primate::fixup_primate_relax), MI.getLoc()));
  }

  // Emit a normal ADD instruction with the given operands.
  MCInst TmpInst = MCInstBuilder(Primate::ADD)
                       .addOperand(DestReg)
                       .addOperand(SrcReg)
                       .addOperand(TPReg);
  uint32_t Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);
}

void PrimateMCCodeEmitter::encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                                              SmallVectorImpl<MCFixup> &Fixups,
                                              const MCSubtargetInfo &STI) const {

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  // Get byte count of instruction.
  unsigned Size = Desc.getSize();

  // PrimateInstrInfo::getInstSizeInBytes hard-codes the number of expanded
  // instructions for each pseudo, and must be updated when adding new pseudos
  // or changing existing ones.
  if (MI.getOpcode() == Primate::PseudoCALLReg ||
      MI.getOpcode() == Primate::PseudoCALL ||
      MI.getOpcode() == Primate::PseudoTAIL ||
      MI.getOpcode() == Primate::PseudoJump) {
    expandFunctionCall(MI, CB, Fixups, STI);
    MCNumEmitted += 2;
    return;
  }

  if (MI.getOpcode() == Primate::PseudoAddTPRel) {
    expandAddTPRel(MI, CB, Fixups, STI);
    MCNumEmitted += 1;
    return;
  }

  // write byte by byte since we don't know the size apriori 
  int mask = 0xff;
  uint64_t inst_bits = getBinaryCodeForInstr(MI, Fixups, STI);
  for(unsigned int i = 0; i < Size; i++) {
    support::endian::write<uint8_t>(CB, inst_bits & mask, llvm::endianness::little);
    inst_bits >>= 8;
  }

  // switch (Size) {
  // default:
  //   llvm_unreachable("Unhandled encodeInstruction length!");
  // case 2: {
  //   uint16_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
  //   support::endian::write<uint16_t>(CB, Bits, llvm::endianness::little);
  //   break;
  // }
  // case 4: {
  //   uint32_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
  //   support::endian::write(CB, Bits, llvm::endianness::little);
  //   break;
  // }
  // case 0:
  //   // Primate: ignore
  //   break;
  // }

  ++MCNumEmitted; // Keep track of the # of mi's emitted.
}

unsigned
PrimateMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {

  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  llvm_unreachable("Unhandled expression!");
  return 0;
}

unsigned
PrimateMCCodeEmitter::getImmOpValueAsr1(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm()) {
    unsigned Res = MO.getImm();
    assert((Res & 1) == 0 && "LSB is non-zero");
    return Res >> 1;
  }

  return getImmOpValue(MI, OpNo, Fixups, STI);
}

unsigned PrimateMCCodeEmitter::getImmOpValue(const MCInst &MI, unsigned OpNo,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  bool EnableRelax = STI.getFeatureBits()[Primate::FeatureRelax];
  const MCOperand &MO = MI.getOperand(OpNo);

  MCInstrDesc const &Desc = MCII.get(MI.getOpcode());
  unsigned MIFrm = PrimateII::getFormat(Desc.TSFlags);

  // If the destination is an immediate, there is nothing to do.
  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr() &&
         "getImmOpValue expects only expressions or immediates");
  const MCExpr *Expr = MO.getExpr();
  MCExpr::ExprKind Kind = Expr->getKind();
  Primate::Fixups FixupKind = Primate::fixup_primate_invalid;
  bool RelaxCandidate = false;
  if (Kind == MCExpr::Target) {
    const PrimateMCExpr *PRExpr = cast<PrimateMCExpr>(Expr);

    switch (PRExpr->getKind()) {
    case PrimateMCExpr::VK_Primate_None:
    case PrimateMCExpr::VK_Primate_Invalid:
    case PrimateMCExpr::VK_Primate_32_PCREL:
      llvm_unreachable("Unhandled fixup kind!");
    case PrimateMCExpr::VK_Primate_TPREL_ADD:
      // tprel_add is only used to indicate that a relocation should be emitted
      // for an add instruction used in TP-relative addressing. It should not be
      // expanded as if representing an actual instruction operand and so to
      // encounter it here is an error.
      llvm_unreachable(
          "VK_Primate_TPREL_ADD should not represent an instruction operand");
    case PrimateMCExpr::VK_Primate_LO:
      if (MIFrm == PrimateII::InstFormatI)
        FixupKind = Primate::fixup_primate_lo12_i;
      else if (MIFrm == PrimateII::InstFormatS)
        FixupKind = Primate::fixup_primate_lo12_s;
      else
        llvm_unreachable("VK_Primate_LO used with unexpected instruction format");
      RelaxCandidate = true;
      break;
    case PrimateMCExpr::VK_Primate_HI:
      FixupKind = Primate::fixup_primate_hi20;
      RelaxCandidate = true;
      break;
    case PrimateMCExpr::VK_Primate_PCREL_LO:
      if (MIFrm == PrimateII::InstFormatI)
        FixupKind = Primate::fixup_primate_pcrel_lo12_i;
      else if (MIFrm == PrimateII::InstFormatS)
        FixupKind = Primate::fixup_primate_pcrel_lo12_s;
      else
        llvm_unreachable(
            "VK_Primate_PCREL_LO used with unexpected instruction format");
      RelaxCandidate = true;
      break;
    case PrimateMCExpr::VK_Primate_PCREL_HI:
      FixupKind = Primate::fixup_primate_pcrel_hi20;
      RelaxCandidate = true;
      break;
    case PrimateMCExpr::VK_Primate_GOT_HI:
      FixupKind = Primate::fixup_primate_got_hi20;
      break;
    case PrimateMCExpr::VK_Primate_TPREL_LO:
      if (MIFrm == PrimateII::InstFormatI)
        FixupKind = Primate::fixup_primate_tprel_lo12_i;
      else if (MIFrm == PrimateII::InstFormatS)
        FixupKind = Primate::fixup_primate_tprel_lo12_s;
      else
        llvm_unreachable(
            "VK_Primate_TPREL_LO used with unexpected instruction format");
      RelaxCandidate = true;
      break;
    case PrimateMCExpr::VK_Primate_TPREL_HI:
      FixupKind = Primate::fixup_primate_tprel_hi20;
      RelaxCandidate = true;
      break;
    case PrimateMCExpr::VK_Primate_TLS_GOT_HI:
      FixupKind = Primate::fixup_primate_tls_got_hi20;
      break;
    case PrimateMCExpr::VK_Primate_TLS_GD_HI:
      FixupKind = Primate::fixup_primate_tls_gd_hi20;
      break;
    case PrimateMCExpr::VK_Primate_CALL:
      FixupKind = Primate::fixup_primate_call;
      RelaxCandidate = true;
      break;
    case PrimateMCExpr::VK_Primate_CALL_PLT:
      FixupKind = Primate::fixup_primate_call_plt;
      RelaxCandidate = true;
      break;
    }
  } else if (Kind == MCExpr::SymbolRef &&
             cast<MCSymbolRefExpr>(Expr)->getKind() == MCSymbolRefExpr::VK_None) {
    if (Desc.getOpcode() == Primate::JAL) {
      FixupKind = Primate::fixup_primate_jal;
    } else if (MIFrm == PrimateII::InstFormatB) {
      FixupKind = Primate::fixup_primate_branch;
    } else if (MIFrm == PrimateII::InstFormatCJ) {
      FixupKind = Primate::fixup_primate_prc_jump;
    } else if (MIFrm == PrimateII::InstFormatCB) {
      FixupKind = Primate::fixup_primate_prc_branch;
    }
  }

  assert(FixupKind != Primate::fixup_primate_invalid && "Unhandled expression!");

  Fixups.push_back(
      MCFixup::create(0, Expr, MCFixupKind(FixupKind), MI.getLoc()));
  ++MCNumFixups;

  // Ensure an R_Primate_RELAX relocation will be emitted if linker relaxation is
  // enabled and the current fixup will result in a relocation that may be
  // relaxed.
  if (EnableRelax && RelaxCandidate) {
    const MCConstantExpr *Dummy = MCConstantExpr::create(0, Ctx);
    Fixups.push_back(
    MCFixup::create(0, Dummy, MCFixupKind(Primate::fixup_primate_relax),
                    MI.getLoc()));
    ++MCNumFixups;
  }

  return 0;
}

unsigned PrimateMCCodeEmitter::getVMaskReg(const MCInst &MI, unsigned OpNo,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  MCOperand MO = MI.getOperand(OpNo);
  assert(MO.isReg() && "Expected a register.");

  switch (MO.getReg()) {
  default:
    llvm_unreachable("Invalid mask register.");
  case Primate::V0:
    return 0;
  case Primate::NoRegister:
    return 1;
  }
}

#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "PrimateGenMCCodeEmitter.inc"
