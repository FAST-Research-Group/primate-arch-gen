//===-- PrimateRegisterBankInfo.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the targeting of the RegisterBankInfo class for Primate.
/// \todo This should be generated by TableGen.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PRIMATE_PRIMATEREGISTERBANKINFO_H
#define LLVM_LIB_TARGET_PRIMATE_PRIMATEREGISTERBANKINFO_H

#include "llvm/CodeGen/RegisterBankInfo.h"

#define GET_REGBANK_DECLARATIONS
#include "PrimateGenRegisterBank.inc"

namespace llvm {

class TargetRegisterInfo;

class PrimateGenRegisterBankInfo : public RegisterBankInfo {
protected:
#define GET_TARGET_REGBANK_CLASS
#include "PrimateGenRegisterBank.inc"
};

/// This class provides the information for the target register banks.
class PrimateRegisterBankInfo final : public PrimateGenRegisterBankInfo {
public:
  PrimateRegisterBankInfo(const TargetRegisterInfo &TRI);
};
} // end namespace llvm
#endif
