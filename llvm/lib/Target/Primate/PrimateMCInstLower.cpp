//===-- PrimateMCInstLower.cpp - Convert Primate MachineInstr to an MCInst ------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Primate MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "Primate.h"
#include "PrimateSubtarget.h"
#include "MCTargetDesc/PrimateMCExpr.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "primate-mcinst-lower"

static MCOperand lowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym,
                                    const AsmPrinter &AP) {
  MCContext &Ctx = AP.OutContext;
  PrimateMCExpr::VariantKind Kind;

  switch (MO.getTargetFlags()) {
  default:
    llvm_unreachable("Unknown target flag on GV operand");
  case PrimateII::MO_None:
    Kind = PrimateMCExpr::VK_Primate_None;
    break;
  case PrimateII::MO_CALL:
    Kind = PrimateMCExpr::VK_Primate_CALL;
    break;
  case PrimateII::MO_PLT:
    Kind = PrimateMCExpr::VK_Primate_CALL_PLT;
    break;
  case PrimateII::MO_LO:
    Kind = PrimateMCExpr::VK_Primate_LO;
    break;
  case PrimateII::MO_HI:
    Kind = PrimateMCExpr::VK_Primate_HI;
    break;
  case PrimateII::MO_PCREL_LO:
    Kind = PrimateMCExpr::VK_Primate_PCREL_LO;
    break;
  case PrimateII::MO_PCREL_HI:
    Kind = PrimateMCExpr::VK_Primate_PCREL_HI;
    break;
  case PrimateII::MO_GOT_HI:
    Kind = PrimateMCExpr::VK_Primate_GOT_HI;
    break;
  case PrimateII::MO_TPREL_LO:
    Kind = PrimateMCExpr::VK_Primate_TPREL_LO;
    break;
  case PrimateII::MO_TPREL_HI:
    Kind = PrimateMCExpr::VK_Primate_TPREL_HI;
    break;
  case PrimateII::MO_TPREL_ADD:
    Kind = PrimateMCExpr::VK_Primate_TPREL_ADD;
    break;
  case PrimateII::MO_TLS_GOT_HI:
    Kind = PrimateMCExpr::VK_Primate_TLS_GOT_HI;
    break;
  case PrimateII::MO_TLS_GD_HI:
    Kind = PrimateMCExpr::VK_Primate_TLS_GD_HI;
    break;
  }

  const MCExpr *ME =
      MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    ME = MCBinaryExpr::createAdd(
        ME, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);

  if (Kind != PrimateMCExpr::VK_Primate_None)
    ME = PrimateMCExpr::create(ME, Kind, Ctx);
  return MCOperand::createExpr(ME);
}

bool llvm::LowerPrimateMachineOperandToMCOperand(const MachineOperand &MO,
                                               MCOperand &MCOp,
                                               const AsmPrinter &AP) {
  switch (MO.getType()) {
  default:
    report_fatal_error("LowerPrimateMachineInstrToMCInst: unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return false;
    MCOp = MCOperand::createReg(MO.getReg());
    break;
  case MachineOperand::MO_RegisterMask:
    // Regmasks are like implicit defs.
    return false;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::createImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = lowerSymbolOperand(MO, MO.getMBB()->getSymbol(), AP);
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = lowerSymbolOperand(MO, AP.getSymbolPreferLocal(*MO.getGlobal()), AP);
    break;
  case MachineOperand::MO_BlockAddress:
    MCOp = lowerSymbolOperand(
        MO, AP.GetBlockAddressSymbol(MO.getBlockAddress()), AP);
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = lowerSymbolOperand(
        MO, AP.GetExternalSymbolSymbol(MO.getSymbolName()), AP);
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = lowerSymbolOperand(MO, AP.GetCPISymbol(MO.getIndex()), AP);
    break;
  case MachineOperand::MO_JumpTableIndex:
    MCOp = lowerSymbolOperand(MO, AP.GetJTISymbol(MO.getIndex()), AP);
    break;
  case MachineOperand::MO_CFIIndex:
    // Primate: ignore
    break;
  case MachineOperand::MO_MCSymbol:
    LLVM_DEBUG(dbgs() << "Lowering MO_MCSymbol\n" << MO.getMCSymbol()->getName() << "\n";);
    MCOp = lowerSymbolOperand(MO, MO.getMCSymbol(), AP);
    break;
  }
  return true;
}

static bool lowerPrimateVMachineInstrToMCInst(const MachineInstr *MI,
                                            MCInst &OutMI) {
  return false;
}

bool llvm::lowerPrimateMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                          AsmPrinter &AP) {
  if (lowerPrimateVMachineInstrToMCInst(MI, OutMI))
    return false;

  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp;
    LLVM_DEBUG(MO.dump(););
    if (LowerPrimateMachineOperandToMCOperand(MO, MCOp, AP))
      OutMI.addOperand(MCOp);
  }

  switch (OutMI.getOpcode()) {
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER: {
    const Function &F = MI->getParent()->getParent()->getFunction();
    if (F.hasFnAttribute("patchable-function-entry")) {
      unsigned Num;
      if (F.getFnAttribute("patchable-function-entry")
              .getValueAsString()
              .getAsInteger(10, Num))
        return false;
      AP.emitNops(Num);
      return true;
    }
    break;
  }
  }
  return false;
}
