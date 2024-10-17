//===-- PrimateISelLowering.cpp - Primate DAG Lowering Implementation  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Primate uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "PrimateISelLowering.h"
#include "MCTargetDesc/PrimateMatInt.h"
#include "Primate.h"
#include "PrimateMachineFunctionInfo.h"
#include "PrimateRegisterInfo.h"
#include "PrimateSubtarget.h"
#include "PrimateTargetMachine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicsPrimate.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "primate-lower"

STATISTIC(NumTailCalls, "Number of tail calls");

PrimateTargetLowering::PrimateTargetLowering(const TargetMachine &TM,
                                         const PrimateSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {

  if (Subtarget.isPRE())
    report_fatal_error("Codegen not yet implemented for PR32E");
    
  PrimateABI::ABI ABI = Subtarget.getTargetABI();
  assert(ABI != PrimateABI::ABI_Unknown && "Improperly initialised target ABI");

  if ((ABI == PrimateABI::ABI_ILP32F || ABI == PrimateABI::ABI_LP64F) &&
      !Subtarget.hasStdExtF()) {
    errs() << "Hard-float 'f' ABI can't be used for a target that "
                "doesn't support the F instruction set extension (ignoring "
                          "target-abi)\n";
    ABI = Subtarget.is64Bit() ? PrimateABI::ABI_LP64 : PrimateABI::ABI_ILP32;
  } else if ((ABI == PrimateABI::ABI_ILP32D || ABI == PrimateABI::ABI_LP64D) &&
             !Subtarget.hasStdExtD()) {
    errs() << "Hard-float 'd' ABI can't be used for a target that "
              "doesn't support the D instruction set extension (ignoring "
              "target-abi)\n";
    ABI = Subtarget.is64Bit() ? PrimateABI::ABI_LP64 : PrimateABI::ABI_ILP32;
  }

  switch (ABI) {
  default:
    report_fatal_error("Don't know how to lower this ABI");
  case PrimateABI::ABI_ILP32:
  case PrimateABI::ABI_ILP32F:
  case PrimateABI::ABI_ILP32D:
  case PrimateABI::ABI_LP64:
  case PrimateABI::ABI_LP64F:
  case PrimateABI::ABI_LP64D:
    break;
  }

  MVT XLenVT = Subtarget.getXLenVT();

  // Set up the register classes.
  addRegisterClass(XLenVT, &Primate::GPRRegClass);

  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Any, LegalizeAction::Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Any, LegalizeAction::Custom);
  addRegisterClass(MVT::Primate_aggregate, &Primate::WIDEREGRegClass);
  addRegisterClass(MVT::i128, &Primate::GPR128RegClass);
  // addRegisterClass(MVT::i8, &Primate::GPR8RegClass);

  //addRegisterClass(MVT::primate_aggre_1, &Primate::WIDEREGRegClass);
  if (Subtarget.hasStdExtZfh())
    addRegisterClass(MVT::f16, &Primate::FPR16RegClass);
  if (Subtarget.hasStdExtF())
    addRegisterClass(MVT::f32, &Primate::FPR32RegClass);
  if (Subtarget.hasStdExtD())
    addRegisterClass(MVT::f64, &Primate::FPR64RegClass);

  static const MVT::SimpleValueType BoolVecVTs[] = {
      MVT::nxv1i1,  MVT::nxv2i1,  MVT::nxv4i1, MVT::nxv8i1,
      MVT::nxv16i1, MVT::nxv32i1, MVT::nxv64i1};
  static const MVT::SimpleValueType IntVecVTs[] = {
      MVT::nxv1i8,  MVT::nxv2i8,   MVT::nxv4i8,   MVT::nxv8i8,  MVT::nxv16i8,
      MVT::nxv32i8, MVT::nxv64i8,  MVT::nxv1i16,  MVT::nxv2i16, MVT::nxv4i16,
      MVT::nxv8i16, MVT::nxv16i16, MVT::nxv32i16, MVT::nxv1i32, MVT::nxv2i32,
      MVT::nxv4i32, MVT::nxv8i32,  MVT::nxv16i32, MVT::nxv1i64, MVT::nxv2i64,
      MVT::nxv4i64, MVT::nxv8i64};
  static const MVT::SimpleValueType F16VecVTs[] = {
      MVT::nxv1f16, MVT::nxv2f16,  MVT::nxv4f16,
      MVT::nxv8f16, MVT::nxv16f16, MVT::nxv32f16};
  static const MVT::SimpleValueType F32VecVTs[] = {
      MVT::nxv1f32, MVT::nxv2f32, MVT::nxv4f32, MVT::nxv8f32, MVT::nxv16f32};
  static const MVT::SimpleValueType F64VecVTs[] = {
      MVT::nxv1f64, MVT::nxv2f64, MVT::nxv4f64, MVT::nxv8f64};

  if (Subtarget.hasStdExtV()) {
    auto addRegClassForPRV = [this](MVT VT) {
      unsigned Size = VT.getSizeInBits().getKnownMinValue();
      assert(Size <= 512 && isPowerOf2_32(Size));
      const TargetRegisterClass *RC;
      if (Size <= 64)
        RC = &Primate::VRRegClass;
      else if (Size == 128)
        RC = &Primate::VRM2RegClass;
      else if (Size == 256)
        RC = &Primate::VRM4RegClass;
      else
        RC = &Primate::VRM8RegClass;

      addRegisterClass(VT, RC);
    };

    for (MVT VT : BoolVecVTs)
      addRegClassForPRV(VT);
    for (MVT VT : IntVecVTs)
      addRegClassForPRV(VT);

    if (Subtarget.hasStdExtZfh())
      for (MVT VT : F16VecVTs)
        addRegClassForPRV(VT);

    if (Subtarget.hasStdExtF())
      for (MVT VT : F32VecVTs)
        addRegClassForPRV(VT);

    if (Subtarget.hasStdExtD())
      for (MVT VT : F64VecVTs)
        addRegClassForPRV(VT);

    if (Subtarget.usePRVForFixedLengthVectors()) {
      auto addRegClassForFixedVectors = [this](MVT VT) {
        MVT ContainerVT = getContainerForFixedLengthVector(VT);
        unsigned RCID = getRegClassIDForVecVT(ContainerVT);
        const PrimateRegisterInfo &TRI = *Subtarget.getRegisterInfo();
        addRegisterClass(VT, TRI.getRegClass(RCID));
      };
      for (MVT VT : MVT::integer_fixedlen_vector_valuetypes())
        if (usePRVForFixedLengthVectorVT(VT))
          addRegClassForFixedVectors(VT);

      for (MVT VT : MVT::fp_fixedlen_vector_valuetypes())
        if (usePRVForFixedLengthVectorVT(VT))
          addRegClassForFixedVectors(VT);
    }
 
  }

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(Primate::X2);

  for (auto N : {ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD})
    setLoadExtAction(N, XLenVT, MVT::i1, Promote);

  // TODO: add all necessary setOperationAction calls.
  setOperationAction(ISD::DYNAMIC_STACKALLOC, XLenVT, Expand);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, XLenVT, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::SELECT_CC, XLenVT, Expand);

  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  if (!Subtarget.hasStdExtZbb()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  }

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::ADD, MVT::i32, Custom);
    setOperationAction(ISD::SUB, MVT::i32, Custom);
    setOperationAction(ISD::MUL, MVT::i32, Custom);
    setOperationAction(ISD::SHL, MVT::i32, Custom);
    setOperationAction(ISD::SRA, MVT::i32, Custom);
    setOperationAction(ISD::SRL, MVT::i32, Custom);

    setOperationAction(ISD::UADDO, MVT::i32, Custom);
    setOperationAction(ISD::USUBO, MVT::i32, Custom);
    setOperationAction(ISD::UADDSAT, MVT::i32, Custom);
    setOperationAction(ISD::USUBSAT, MVT::i32, Custom);
  }

  if (!Subtarget.hasStdExtM()) {
    // setOperationAction(ISD::MUL, XLenVT, Expand);
    // setOperationAction(ISD::MULHS, XLenVT, Expand);
    // setOperationAction(ISD::MULHU, XLenVT, Expand);
    // setOperationAction(ISD::SDIV, XLenVT, Expand);
    // setOperationAction(ISD::UDIV, XLenVT, Expand);
    // setOperationAction(ISD::SREM, XLenVT, Expand);
    // setOperationAction(ISD::UREM, XLenVT, Expand);
  } else {
    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::MUL, MVT::i32, Custom);
      setOperationAction(ISD::MUL, MVT::i128, Custom);

      setOperationAction(ISD::SDIV, MVT::i8, Custom);
      setOperationAction(ISD::UDIV, MVT::i8, Custom);
      setOperationAction(ISD::UREM, MVT::i8, Custom);
      setOperationAction(ISD::SDIV, MVT::i16, Custom);
      setOperationAction(ISD::UDIV, MVT::i16, Custom);
      setOperationAction(ISD::UREM, MVT::i16, Custom);
      setOperationAction(ISD::SDIV, MVT::i32, Custom);
      setOperationAction(ISD::UDIV, MVT::i32, Custom);
      setOperationAction(ISD::UREM, MVT::i32, Custom);
    } else {
      setOperationAction(ISD::MUL, MVT::i64, Custom);
    }
  }

  setOperationAction(ISD::SDIVREM, XLenVT, Expand);
  setOperationAction(ISD::UDIVREM, XLenVT, Expand);
  setOperationAction(ISD::SMUL_LOHI, XLenVT, Expand);
  setOperationAction(ISD::UMUL_LOHI, XLenVT, Expand);

  setOperationAction(ISD::SHL_PARTS, XLenVT, Custom);
  setOperationAction(ISD::SRL_PARTS, XLenVT, Custom);
  setOperationAction(ISD::SRA_PARTS, XLenVT, Custom);

  if (Subtarget.hasStdExtZbb()) {
    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::ROTL, MVT::i32, Custom);
      setOperationAction(ISD::ROTR, MVT::i32, Custom);
    }
  } else {
    setOperationAction(ISD::ROTL, XLenVT, Expand);
    setOperationAction(ISD::ROTR, XLenVT, Expand);
  }

  if (Subtarget.hasStdExtZbb()) {
    // Custom lower bswap/bitreverse so we can convert them to GREVI to enable
    // more combining.
    setOperationAction(ISD::BITREVERSE, XLenVT,   Custom);
    setOperationAction(ISD::BSWAP,      XLenVT,   Custom);
    setOperationAction(ISD::BITREVERSE, MVT::i8,  Custom);
    // BSWAP i8 doesn't exist.
    setOperationAction(ISD::BITREVERSE, MVT::i16, Custom);
    setOperationAction(ISD::BSWAP,      MVT::i16, Custom);

    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::BITREVERSE, MVT::i32, Custom);
      setOperationAction(ISD::BSWAP,      MVT::i32, Custom);
    }
  } else {
    // With Zbb we have an XLen rev8 instruction, but not GREVI. So we'll
    // pattern match it directly in isel.
    setOperationAction(ISD::BSWAP, XLenVT,
                       Subtarget.hasStdExtZbb() ? Legal : Expand);
  }

  if (Subtarget.hasStdExtZbb()) {
    setOperationAction(ISD::SMIN, XLenVT, Legal);
    setOperationAction(ISD::SMAX, XLenVT, Legal);
    setOperationAction(ISD::UMIN, XLenVT, Legal);
    setOperationAction(ISD::UMAX, XLenVT, Legal);

    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::CTTZ, MVT::i32, Custom);
      setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Custom);
      setOperationAction(ISD::CTLZ, MVT::i32, Custom);
      setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Custom);
    }
  } else {
    setOperationAction(ISD::CTTZ, XLenVT, Expand);
    setOperationAction(ISD::CTLZ, XLenVT, Expand);
    setOperationAction(ISD::CTPOP, XLenVT, Expand);
  }

  if (Subtarget.hasStdExtZbb()) {
    setOperationAction(ISD::FSHL, XLenVT, Custom);
    setOperationAction(ISD::FSHR, XLenVT, Custom);
    setOperationAction(ISD::SELECT, XLenVT, Legal);

    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::FSHL, MVT::i32, Custom);
      setOperationAction(ISD::FSHR, MVT::i32, Custom);
    }
  } else {
    setOperationAction(ISD::SELECT, XLenVT, Custom);
  }

  ISD::CondCode FPCCToExpand[] = {
      ISD::SETOGT, ISD::SETOGE, ISD::SETONE, ISD::SETUEQ, ISD::SETUGT,
      ISD::SETUGE, ISD::SETULT, ISD::SETULE, ISD::SETUNE, ISD::SETGT,
      ISD::SETGE,  ISD::SETNE,  ISD::SETO,   ISD::SETUO};

  ISD::NodeType FPOpToExpand[] = {
      ISD::FSIN, ISD::FCOS, ISD::FSINCOS, ISD::FPOW, ISD::FREM, ISD::FP16_TO_FP,
      ISD::FP_TO_FP16};

  if (Subtarget.hasStdExtZfh())
    setOperationAction(ISD::BITCAST, MVT::i16, Custom);

  if (Subtarget.hasStdExtZfh()) {
    setOperationAction(ISD::FMINNUM, MVT::f16, Legal);
    setOperationAction(ISD::FMAXNUM, MVT::f16, Legal);
    setOperationAction(ISD::LRINT, MVT::f16, Legal);
    setOperationAction(ISD::LLRINT, MVT::f16, Legal);
    setOperationAction(ISD::LROUND, MVT::f16, Legal);
    setOperationAction(ISD::LLROUND, MVT::f16, Legal);
    for (auto CC : FPCCToExpand)
      setCondCodeAction(CC, MVT::f16, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f16, Expand);
    setOperationAction(ISD::SELECT, MVT::f16, Custom);
    setOperationAction(ISD::BR_CC, MVT::f16, Expand);
    for (auto Op : FPOpToExpand)
      setOperationAction(Op, MVT::f16, Expand);
  }

  if (Subtarget.hasStdExtF()) {
    setOperationAction(ISD::FMINNUM, MVT::f32, Legal);
    setOperationAction(ISD::FMAXNUM, MVT::f32, Legal);
    setOperationAction(ISD::LRINT, MVT::f32, Legal);
    setOperationAction(ISD::LLRINT, MVT::f32, Legal);
    setOperationAction(ISD::LROUND, MVT::f32, Legal);
    setOperationAction(ISD::LLROUND, MVT::f32, Legal);
    for (auto CC : FPCCToExpand)
      setCondCodeAction(CC, MVT::f32, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
    setOperationAction(ISD::SELECT, MVT::f32, Custom);
    setOperationAction(ISD::BR_CC, MVT::f32, Expand);
    for (auto Op : FPOpToExpand)
      setOperationAction(Op, MVT::f32, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f16, Expand);
    setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  }

  if (Subtarget.hasStdExtF() && Subtarget.is64Bit())
    setOperationAction(ISD::BITCAST, MVT::i32, Custom);

  if (Subtarget.hasStdExtD()) {
    setOperationAction(ISD::FMINNUM, MVT::f64, Legal);
    setOperationAction(ISD::FMAXNUM, MVT::f64, Legal);
    setOperationAction(ISD::LRINT, MVT::f64, Legal);
    setOperationAction(ISD::LLRINT, MVT::f64, Legal);
    setOperationAction(ISD::LROUND, MVT::f64, Legal);
    setOperationAction(ISD::LLROUND, MVT::f64, Legal);
    for (auto CC : FPCCToExpand)
      setCondCodeAction(CC, MVT::f64, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
    setOperationAction(ISD::SELECT, MVT::f64, Custom);
    setOperationAction(ISD::BR_CC, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
    for (auto Op : FPOpToExpand)
      setOperationAction(Op, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f16, Expand);
    setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  }

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
    setOperationAction(ISD::STRICT_FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::STRICT_FP_TO_SINT, MVT::i32, Custom);
  }

  if (Subtarget.hasStdExtF()) {
    setOperationAction(ISD::FP_ROUND, XLenVT, Custom);
    setOperationAction(ISD::SET_ROUNDING, MVT::Other, Custom);
  }

  setOperationAction(ISD::GlobalAddress, XLenVT, Custom);
  setOperationAction(ISD::BlockAddress, XLenVT, Custom);
  setOperationAction(ISD::ConstantPool, XLenVT, Custom);
  setOperationAction(ISD::JumpTable, XLenVT, Custom);

  setOperationAction(ISD::GlobalTLSAddress, XLenVT, Custom);

  // TODO: On M-mode only targets, the cycle[h] CSR may not be present.
  // Unfortunately this can't be determined just from the ISA naming string.
  setOperationAction(ISD::READCYCLECOUNTER, MVT::i64,
                     Subtarget.is64Bit() ? Legal : Custom);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  if (Subtarget.is64Bit())
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i32, Custom);

  if (Subtarget.hasStdExtA()) {
    setMaxAtomicSizeInBitsSupported(Subtarget.getXLen());
    setMinCmpXchgSizeInBits(32);
  } else {
    setMaxAtomicSizeInBitsSupported(0);
  }

  setBooleanContents(ZeroOrOneBooleanContent);

  if (Subtarget.hasStdExtV()) {
    setBooleanVectorContents(ZeroOrOneBooleanContent);

    setOperationAction(ISD::VSCALE, XLenVT, Custom);

    // PRV intrinsics may have illegal operands.
    // We also need to custom legalize vmv.x.s.
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i8, Custom);
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i16, Custom);
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i8, Custom);
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i16, Custom);
    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i32, Custom);
    } else {
      setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i64, Custom);
      setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i64, Custom);
    }

    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);

    static unsigned IntegerVPOps[] = {
        ISD::VP_ADD,  ISD::VP_SUB,  ISD::VP_MUL, ISD::VP_SDIV, ISD::VP_UDIV,
        ISD::VP_SREM, ISD::VP_UREM, ISD::VP_AND, ISD::VP_OR,   ISD::VP_XOR,
        ISD::VP_ASHR, ISD::VP_LSHR, ISD::VP_SHL};

    static unsigned FloatingPointVPOps[] = {ISD::VP_FADD, ISD::VP_FSUB,
                                            ISD::VP_FMUL, ISD::VP_FDIV};

    if (!Subtarget.is64Bit()) {
      // We must custom-lower certain vXi64 operations on PR32 due to the vector
      // element type being illegal.
      setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::i64, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::i64, Custom);

      setOperationAction(ISD::VECREDUCE_ADD, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_AND, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_OR, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_XOR, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_SMAX, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_SMIN, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_UMAX, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_UMIN, MVT::i64, Custom);
    }

    for (MVT VT : BoolVecVTs) {
      setOperationAction(ISD::SPLAT_VECTOR, VT, Custom);

      // Mask VTs are custom-expanded into a series of standard nodes
      setOperationAction(ISD::TRUNCATE, VT, Custom);
      setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);
      setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
      setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

      setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);

      setOperationAction(ISD::SELECT, VT, Custom);
      setOperationAction(ISD::SELECT_CC, VT, Expand);
      setOperationAction(ISD::VSELECT, VT, Expand);

      setOperationAction(ISD::VECREDUCE_AND, VT, Custom);
      setOperationAction(ISD::VECREDUCE_OR, VT, Custom);
      setOperationAction(ISD::VECREDUCE_XOR, VT, Custom);

      // PRV has native int->float & float->int conversions where the
      // element type sizes are within one power-of-two of each other. Any
      // wider distances between type sizes have to be lowered as sequences
      // which progressively narrow the gap in stages.
      setOperationAction(ISD::SINT_TO_FP, VT, Custom);
      setOperationAction(ISD::UINT_TO_FP, VT, Custom);
      setOperationAction(ISD::FP_TO_SINT, VT, Custom);
      setOperationAction(ISD::FP_TO_UINT, VT, Custom);

      // Expand all extending loads to types larger than this, and truncating
      // stores from types larger than this.
      for (MVT OtherVT : MVT::integer_scalable_vector_valuetypes()) {
        setTruncStoreAction(OtherVT, VT, Expand);
        setLoadExtAction(ISD::EXTLOAD, OtherVT, VT, Expand);
        setLoadExtAction(ISD::SEXTLOAD, OtherVT, VT, Expand);
        setLoadExtAction(ISD::ZEXTLOAD, OtherVT, VT, Expand);
      }
    }

    for (MVT VT : IntVecVTs) {
      setOperationAction(ISD::SPLAT_VECTOR, VT, Legal);
      setOperationAction(ISD::SPLAT_VECTOR_PARTS, VT, Custom);

      setOperationAction(ISD::SMIN, VT, Legal);
      setOperationAction(ISD::SMAX, VT, Legal);
      setOperationAction(ISD::UMIN, VT, Legal);
      setOperationAction(ISD::UMAX, VT, Legal);

      setOperationAction(ISD::ROTL, VT, Expand);
      setOperationAction(ISD::ROTR, VT, Expand);

      // Custom-lower extensions and truncations from/to mask types.
      setOperationAction(ISD::ANY_EXTEND, VT, Custom);
      setOperationAction(ISD::SIGN_EXTEND, VT, Custom);
      setOperationAction(ISD::ZERO_EXTEND, VT, Custom);

      // PRV has native int->float & float->int conversions where the
      // element type sizes are within one power-of-two of each other. Any
      // wider distances between type sizes have to be lowered as sequences
      // which progressively narrow the gap in stages.
      setOperationAction(ISD::SINT_TO_FP, VT, Custom);
      setOperationAction(ISD::UINT_TO_FP, VT, Custom);
      setOperationAction(ISD::FP_TO_SINT, VT, Custom);
      setOperationAction(ISD::FP_TO_UINT, VT, Custom);

      setOperationAction(ISD::SADDSAT, VT, Legal);
      setOperationAction(ISD::UADDSAT, VT, Legal);
      setOperationAction(ISD::SSUBSAT, VT, Legal);
      setOperationAction(ISD::USUBSAT, VT, Legal);

      // Integer VTs are lowered as a series of "PrimateISD::TRUNCATE_VECTOR_VL"
      // nodes which truncate by one power of two at a time.
      setOperationAction(ISD::TRUNCATE, VT, Custom);

      // Custom-lower insert/extract operations to simplify patterns.
      setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);

      // Custom-lower reduction operations to set up the corresponding custom
      // nodes' operands.
      setOperationAction(ISD::VECREDUCE_ADD, VT, Custom);
      setOperationAction(ISD::VECREDUCE_AND, VT, Custom);
      setOperationAction(ISD::VECREDUCE_OR, VT, Custom);
      setOperationAction(ISD::VECREDUCE_XOR, VT, Custom);
      setOperationAction(ISD::VECREDUCE_SMAX, VT, Custom);
      setOperationAction(ISD::VECREDUCE_SMIN, VT, Custom);
      setOperationAction(ISD::VECREDUCE_UMAX, VT, Custom);
      setOperationAction(ISD::VECREDUCE_UMIN, VT, Custom);

      for (unsigned VPOpc : IntegerVPOps)
        setOperationAction(VPOpc, VT, Custom);

      setOperationAction(ISD::LOAD, VT, Custom);
      setOperationAction(ISD::STORE, VT, Custom);

      setOperationAction(ISD::MLOAD, VT, Custom);
      setOperationAction(ISD::MSTORE, VT, Custom);
      setOperationAction(ISD::MGATHER, VT, Custom);
      setOperationAction(ISD::MSCATTER, VT, Custom);

      setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);
      setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
      setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

      setOperationAction(ISD::SELECT, VT, Custom);
      setOperationAction(ISD::SELECT_CC, VT, Expand);

      setOperationAction(ISD::STEP_VECTOR, VT, Custom);
      setOperationAction(ISD::VECTOR_REVERSE, VT, Custom);

      for (MVT OtherVT : MVT::integer_scalable_vector_valuetypes()) {
        setTruncStoreAction(VT, OtherVT, Expand);
        setLoadExtAction(ISD::EXTLOAD, OtherVT, VT, Expand);
        setLoadExtAction(ISD::SEXTLOAD, OtherVT, VT, Expand);
        setLoadExtAction(ISD::ZEXTLOAD, OtherVT, VT, Expand);
      }
    }

    // Expand various CCs to best match the PRV ISA, which natively supports UNE
    // but no other unordered comparisons, and supports all ordered comparisons
    // except ONE. Additionally, we expand GT,OGT,GE,OGE for optimization
    // purposes; they are expanded to their swapped-operand CCs (LT,OLT,LE,OLE),
    // and we pattern-match those back to the "original", swapping operands once
    // more. This way we catch both operations and both "vf" and "fv" forms with
    // fewer patterns.
    ISD::CondCode VFPCCToExpand[] = {
        ISD::SETO,   ISD::SETONE, ISD::SETUEQ, ISD::SETUGT,
        ISD::SETUGE, ISD::SETULT, ISD::SETULE, ISD::SETUO,
        ISD::SETGT,  ISD::SETOGT, ISD::SETGE,  ISD::SETOGE,
    };

    // Sets common operation actions on PRV floating-point vector types.
    const auto SetCommonVFPActions = [&](MVT VT) {
      setOperationAction(ISD::SPLAT_VECTOR, VT, Legal);
      // PRV has native FP_ROUND & FP_EXTEND conversions where the element type
      // sizes are within one power-of-two of each other. Therefore conversions
      // between vXf16 and vXf64 must be lowered as sequences which convert via
      // vXf32.
      setOperationAction(ISD::FP_ROUND, VT, Custom);
      setOperationAction(ISD::FP_EXTEND, VT, Custom);
      // Custom-lower insert/extract operations to simplify patterns.
      setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
      // Expand various condition codes (explained above).
      for (auto CC : VFPCCToExpand)
        setCondCodeAction(CC, VT, Expand);

      setOperationAction(ISD::FMINNUM, VT, Legal);
      setOperationAction(ISD::FMAXNUM, VT, Legal);

      setOperationAction(ISD::VECREDUCE_FADD, VT, Custom);
      setOperationAction(ISD::VECREDUCE_SEQ_FADD, VT, Custom);
      setOperationAction(ISD::VECREDUCE_FMIN, VT, Custom);
      setOperationAction(ISD::VECREDUCE_FMAX, VT, Custom);
      setOperationAction(ISD::FCOPYSIGN, VT, Legal);

      setOperationAction(ISD::LOAD, VT, Custom);
      setOperationAction(ISD::STORE, VT, Custom);

      setOperationAction(ISD::MLOAD, VT, Custom);
      setOperationAction(ISD::MSTORE, VT, Custom);
      setOperationAction(ISD::MGATHER, VT, Custom);
      setOperationAction(ISD::MSCATTER, VT, Custom);

      setOperationAction(ISD::SELECT, VT, Custom);
      setOperationAction(ISD::SELECT_CC, VT, Expand);

      setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);
      setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
      setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

      setOperationAction(ISD::VECTOR_REVERSE, VT, Custom);

      for (unsigned VPOpc : FloatingPointVPOps)
        setOperationAction(VPOpc, VT, Custom);
    };

    // Sets common extload/truncstore actions on PRV floating-point vector
    // types.
    const auto SetCommonVFPExtLoadTruncStoreActions =
        [&](MVT VT, ArrayRef<MVT::SimpleValueType> SmallerVTs) {
          for (auto SmallVT : SmallerVTs) {
            setTruncStoreAction(VT, SmallVT, Expand);
            setLoadExtAction(ISD::EXTLOAD, VT, SmallVT, Expand);
          }
        };

    if (Subtarget.hasStdExtZfh())
      for (MVT VT : F16VecVTs)
        SetCommonVFPActions(VT);

    for (MVT VT : F32VecVTs) {
      if (Subtarget.hasStdExtF())
        SetCommonVFPActions(VT);
      SetCommonVFPExtLoadTruncStoreActions(VT, F16VecVTs);
    }

    for (MVT VT : F64VecVTs) {
      if (Subtarget.hasStdExtD())
        SetCommonVFPActions(VT);
      SetCommonVFPExtLoadTruncStoreActions(VT, F16VecVTs);
      SetCommonVFPExtLoadTruncStoreActions(VT, F32VecVTs);
    }

    if (Subtarget.usePRVForFixedLengthVectors()) {
      for (MVT VT : MVT::integer_fixedlen_vector_valuetypes()) {
        if (!usePRVForFixedLengthVectorVT(VT))
          continue;

        // By default everything must be expanded.
        for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op)
          setOperationAction(Op, VT, Expand);
        for (MVT OtherVT : MVT::integer_fixedlen_vector_valuetypes()) {
          setTruncStoreAction(VT, OtherVT, Expand);
          setLoadExtAction(ISD::EXTLOAD, OtherVT, VT, Expand);
          setLoadExtAction(ISD::SEXTLOAD, OtherVT, VT, Expand);
          setLoadExtAction(ISD::ZEXTLOAD, OtherVT, VT, Expand);
        }

        // We use EXTRACT_SUBVECTOR as a "cast" from scalable to fixed.
        setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
        setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

        setOperationAction(ISD::BUILD_VECTOR, VT, Custom);
        setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);

        setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
        setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);

        setOperationAction(ISD::LOAD, VT, Custom);
        setOperationAction(ISD::STORE, VT, Custom);

        setOperationAction(ISD::SETCC, VT, Custom);

        setOperationAction(ISD::SELECT, VT, Custom);

        setOperationAction(ISD::TRUNCATE, VT, Custom);

        setOperationAction(ISD::BITCAST, VT, Custom);

        setOperationAction(ISD::VECREDUCE_AND, VT, Custom);
        setOperationAction(ISD::VECREDUCE_OR, VT, Custom);
        setOperationAction(ISD::VECREDUCE_XOR, VT, Custom);

        setOperationAction(ISD::SINT_TO_FP, VT, Custom);
        setOperationAction(ISD::UINT_TO_FP, VT, Custom);
        setOperationAction(ISD::FP_TO_SINT, VT, Custom);
        setOperationAction(ISD::FP_TO_UINT, VT, Custom);

        // Operations below are different for between masks and other vectors.
        if (VT.getVectorElementType() == MVT::i1) {
          setOperationAction(ISD::AND, VT, Custom);
          setOperationAction(ISD::OR, VT, Custom);
          setOperationAction(ISD::XOR, VT, Custom);
          continue;
        }

        // Use SPLAT_VECTOR to prevent type legalization from destroying the
        // splats when type legalizing i64 scalar on PR32.
        // FIXME: Use SPLAT_VECTOR for all types? DAGCombine probably needs
        // improvements first.
        if (!Subtarget.is64Bit() && VT.getVectorElementType() == MVT::i64) {
          setOperationAction(ISD::SPLAT_VECTOR, VT, Custom);
          setOperationAction(ISD::SPLAT_VECTOR_PARTS, VT, Custom);
        }

        setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
        setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);

        setOperationAction(ISD::MLOAD, VT, Custom);
        setOperationAction(ISD::MSTORE, VT, Custom);
        setOperationAction(ISD::MGATHER, VT, Custom);
        setOperationAction(ISD::MSCATTER, VT, Custom);
        setOperationAction(ISD::ADD, VT, Custom);
        setOperationAction(ISD::MUL, VT, Custom);
        setOperationAction(ISD::SUB, VT, Custom);
        setOperationAction(ISD::AND, VT, Custom);
        setOperationAction(ISD::OR, VT, Custom);
        setOperationAction(ISD::XOR, VT, Custom);
        setOperationAction(ISD::SDIV, VT, Custom);
        setOperationAction(ISD::SREM, VT, Custom);
        setOperationAction(ISD::UDIV, VT, Custom);
        setOperationAction(ISD::UREM, VT, Custom);
        setOperationAction(ISD::SHL, VT, Custom);
        setOperationAction(ISD::SRA, VT, Custom);
        setOperationAction(ISD::SRL, VT, Custom);

        setOperationAction(ISD::SMIN, VT, Custom);
        setOperationAction(ISD::SMAX, VT, Custom);
        setOperationAction(ISD::UMIN, VT, Custom);
        setOperationAction(ISD::UMAX, VT, Custom);
        setOperationAction(ISD::ABS,  VT, Custom);

        setOperationAction(ISD::MULHS, VT, Custom);
        setOperationAction(ISD::MULHU, VT, Custom);

        setOperationAction(ISD::SADDSAT, VT, Custom);
        setOperationAction(ISD::UADDSAT, VT, Custom);
        setOperationAction(ISD::SSUBSAT, VT, Custom);
        setOperationAction(ISD::USUBSAT, VT, Custom);

        setOperationAction(ISD::VSELECT, VT, Custom);
        setOperationAction(ISD::SELECT_CC, VT, Expand);

        setOperationAction(ISD::ANY_EXTEND, VT, Custom);
        setOperationAction(ISD::SIGN_EXTEND, VT, Custom);
        setOperationAction(ISD::ZERO_EXTEND, VT, Custom);

        // Custom-lower reduction operations to set up the corresponding custom
        // nodes' operands.
        setOperationAction(ISD::VECREDUCE_ADD, VT, Custom);
        setOperationAction(ISD::VECREDUCE_SMAX, VT, Custom);
        setOperationAction(ISD::VECREDUCE_SMIN, VT, Custom);
        setOperationAction(ISD::VECREDUCE_UMAX, VT, Custom);
        setOperationAction(ISD::VECREDUCE_UMIN, VT, Custom);

        for (unsigned VPOpc : IntegerVPOps)
          setOperationAction(VPOpc, VT, Custom);
      }

      for (MVT VT : MVT::fp_fixedlen_vector_valuetypes()) {
        if (!usePRVForFixedLengthVectorVT(VT))
          continue;

        // By default everything must be expanded.
        for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op)
          setOperationAction(Op, VT, Expand);
        for (MVT OtherVT : MVT::fp_fixedlen_vector_valuetypes()) {
          setLoadExtAction(ISD::EXTLOAD, OtherVT, VT, Expand);
          setTruncStoreAction(VT, OtherVT, Expand);
        }

        // We use EXTRACT_SUBVECTOR as a "cast" from scalable to fixed.
        setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
        setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

        setOperationAction(ISD::BUILD_VECTOR, VT, Custom);
        setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
        setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
        setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);

        setOperationAction(ISD::LOAD, VT, Custom);
        setOperationAction(ISD::STORE, VT, Custom);
        setOperationAction(ISD::MLOAD, VT, Custom);
        setOperationAction(ISD::MSTORE, VT, Custom);
        setOperationAction(ISD::MGATHER, VT, Custom);
        setOperationAction(ISD::MSCATTER, VT, Custom);
        setOperationAction(ISD::FADD, VT, Custom);
        setOperationAction(ISD::FSUB, VT, Custom);
        setOperationAction(ISD::FMUL, VT, Custom);
        setOperationAction(ISD::FDIV, VT, Custom);
        setOperationAction(ISD::FNEG, VT, Custom);
        setOperationAction(ISD::FABS, VT, Custom);
        setOperationAction(ISD::FCOPYSIGN, VT, Custom);
        setOperationAction(ISD::FSQRT, VT, Custom);
        setOperationAction(ISD::FMA, VT, Custom);
        setOperationAction(ISD::FMINNUM, VT, Custom);
        setOperationAction(ISD::FMAXNUM, VT, Custom);

        setOperationAction(ISD::FP_ROUND, VT, Custom);
        setOperationAction(ISD::FP_EXTEND, VT, Custom);

        for (auto CC : VFPCCToExpand)
          setCondCodeAction(CC, VT, Expand);

        setOperationAction(ISD::VSELECT, VT, Custom);
        setOperationAction(ISD::SELECT, VT, Custom);
        setOperationAction(ISD::SELECT_CC, VT, Expand);

        setOperationAction(ISD::BITCAST, VT, Custom);

        setOperationAction(ISD::VECREDUCE_FADD, VT, Custom);
        setOperationAction(ISD::VECREDUCE_SEQ_FADD, VT, Custom);
        setOperationAction(ISD::VECREDUCE_FMIN, VT, Custom);
        setOperationAction(ISD::VECREDUCE_FMAX, VT, Custom);

        for (unsigned VPOpc : FloatingPointVPOps)
          setOperationAction(VPOpc, VT, Custom);
      }

      // Custom-legalize bitcasts from fixed-length vectors to scalar types.
      setOperationAction(ISD::BITCAST, MVT::i8, Custom);
      setOperationAction(ISD::BITCAST, MVT::i16, Custom);
      setOperationAction(ISD::BITCAST, MVT::i32, Custom);
      setOperationAction(ISD::BITCAST, MVT::i64, Custom);
      setOperationAction(ISD::BITCAST, MVT::f16, Custom);
      setOperationAction(ISD::BITCAST, MVT::f32, Custom);
      setOperationAction(ISD::BITCAST, MVT::f64, Custom);
    }
  }

  // Function alignment set to 1 since we do not care about alignment.
  const Align FunctionAlignment(1);
  setMinFunctionAlignment(FunctionAlignment);
  setPrefFunctionAlignment(FunctionAlignment);

  setMinimumJumpTableEntries(5);

  // Jumps are expensive, compared to logic
  setJumpIsExpensive();

  // We can use any register for comparisons
  setHasMultipleConditionRegisters();

  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::XOR);
  setTargetDAGCombine(ISD::ANY_EXTEND);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  if (Subtarget.hasStdExtV()) {
    setTargetDAGCombine(ISD::FCOPYSIGN);
    setTargetDAGCombine(ISD::MGATHER);
    setTargetDAGCombine(ISD::MSCATTER);
    setTargetDAGCombine(ISD::SRA);
    setTargetDAGCombine(ISD::SRL);
    setTargetDAGCombine(ISD::SHL);
  }
  int alucount = 0;
  int bfucount = 0;
  // Need to read in the archgen params for the reg file.
  dbgs() << "reading in register indexing parameters\n";
  std::ifstream archgenParams ("primate.cfg");
  if(!archgenParams.good()) {
    errs() << "primate.cfg not found! any default we try will be bad. (Run arch-gen?)\n";
    errs() << "This better not run the backend!\n";
  }
  else {
    while(archgenParams.good()) {
      std::string paramString;
      getline(archgenParams, paramString);
      std::string name  = paramString.substr(0, paramString.find("=")); 
      std::string value = paramString.substr(paramString.find("=")+1, paramString.length());
      if(name == "SRC_POS") {
	dbgs() << value << "\n";
	auto iss = std::istringstream{value};
	auto str = std::string{};

	while (iss >> str) {
	  allPoses.push_back(std::stoi(str));
	}
      }
      else if(name == "SRC_MODE") {
	dbgs() << value << "\n";
	auto iss = std::istringstream{value};
	auto str = std::string{};

	while (iss >> str) {
	  allSizes.push_back(std::stoi(str));
	}
      }
      else if(name == "NUM_ALUS") {
	alucount = std::stoi(value);
	dbgs() << "number of ALUs found: " << alucount << "\n";
      }
      else if(name == "NUM_BFUS") {
	bfucount = std::stoi(value) + 2;
	dbgs() << "number of BFUs found: " << bfucount << "\n";
      }
    }
  }

  int functionalUnitIdx = 0;
  int slotIdx = 0;
  while(true) {
    if (alucount > 0 && bfucount > 0) {
      dbgs() << "Merged slot idx: " << functionalUnitIdx << "\n";
      alucount--;
      bfucount--;
      allSlotInfo.push_back(SlotTypes::EXTRACT);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
      allSlotInfo.push_back(SlotTypes::EXTRACT);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
      allSlotInfo.push_back(SlotTypes::MERGED);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
      allSlotInfo.push_back(SlotTypes::INSERT);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
      
    }
    else if(alucount > 0) {
      dbgs() << "ALU slot idx: " << functionalUnitIdx << "\n";
      alucount--;
      allSlotInfo.push_back(SlotTypes::EXTRACT);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
      allSlotInfo.push_back(SlotTypes::EXTRACT);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
      allSlotInfo.push_back(SlotTypes::GREEN);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
      allSlotInfo.push_back(SlotTypes::INSERT);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
    }
    else if(bfucount > 0) {
      dbgs() << "BFU slot idx: " << functionalUnitIdx << "\n";
      bfucount--;
      allSlotInfo.push_back(SlotTypes::BLUE);
      slotToFUIndex[slotIdx] = functionalUnitIdx;
      slotIdx++;
    }
    else if(alucount == 0 && bfucount == 0) {
      break;
    }
    functionalUnitIdx++;
  }
  allSlotInfo.push_back(SlotTypes::BRANCH);
  slotToFUIndex[slotIdx] = functionalUnitIdx;
}

bool PrimateTargetLowering::supportedArray(ArrayType &ATy, int bitpos) const {
  uint64_t ele = ATy.getNumElements();
  auto eTy = ATy.getElementType();
  if(!(eTy->isSized())) {
    llvm_unreachable("struct contains elements that are unsized types");
  }
  // array types need to access individual elements
  if(auto aty = dyn_cast<ArrayType>(eTy)) {
    if(!supportedArray(*aty, bitpos))
      return false;
  }
  else if(auto sty = dyn_cast<StructType>(eTy)) {
    if(!supportedAggregate(*sty, bitpos))
      return false;
  }
  else {
    if(find(allSizes.begin(), allSizes.end(), eTy->getScalarSizeInBits()) == allSizes.end()) {
      LLVM_DEBUG(dbgs() << "array failed to match regs due to element size unsupported\n";);
      return false;
    }
    if(find(allPoses.begin(), allPoses.end(), bitpos) == allPoses.end()) {
      LLVM_DEBUG(dbgs() << "array failed to match regs due to element offset unsupported\n";);
      return false;
    }
  }
  bitpos += eTy->getScalarSizeInBits() * ele;
  return true;
}

bool PrimateTargetLowering::supportedAggregate(StructType &STy, int bitpos) const {
  for(Type* eTy: STy.elements()) {
    if(!(eTy->isSized())) {
      llvm_unreachable("struct contains elements that are unsized types");
    }
    // array types need to access individual elements
    if(auto aty = dyn_cast<ArrayType>(eTy)) {
      if(!supportedArray(*aty, bitpos))
        return false;
    }
    else if(auto sty = dyn_cast<StructType>(eTy)) {
      if(!supportedAggregate(*sty, bitpos))
        return false;
    }
    else {
      if(find(allSizes.begin(), allSizes.end(), eTy->getScalarSizeInBits()) == allSizes.end()) {
        LLVM_DEBUG(dbgs() << "struct failed to match regs due to element size unsupported\n";);
        return false;
      }
      if(find(allPoses.begin(), allPoses.end(), bitpos) == allPoses.end()) {
        LLVM_DEBUG(dbgs() << "struct failed to match regs due to element offset unsupported\n";);
        return false;
      }
    }
    bitpos += eTy->getScalarSizeInBits();
  }
  return true;
}

EVT PrimateTargetLowering::getSetCCResultType(const DataLayout &DL,
                                            LLVMContext &Context,
                                            EVT VT) const {
  if (!VT.isVector())
    return getPointerTy(DL);
  if (Subtarget.hasStdExtV() &&
      (VT.isScalableVector() || Subtarget.usePRVForFixedLengthVectors()))
    return EVT::getVectorVT(Context, MVT::i1, VT.getVectorElementCount());
  return VT.changeVectorElementTypeToInteger();
}

MVT PrimateTargetLowering::getVPExplicitVectorLengthTy() const {
  return Subtarget.getXLenVT();
}

bool PrimateTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                             const CallInst &I,
                                             MachineFunction &MF,
                                             unsigned Intrinsic) const {
  switch (Intrinsic) {
  default:
    return false;
  case Intrinsic::primate_masked_atomicrmw_xchg_i32:
  case Intrinsic::primate_masked_atomicrmw_add_i32:
  case Intrinsic::primate_masked_atomicrmw_sub_i32:
  case Intrinsic::primate_masked_atomicrmw_nand_i32:
  case Intrinsic::primate_masked_atomicrmw_max_i32:
  case Intrinsic::primate_masked_atomicrmw_min_i32:
  case Intrinsic::primate_masked_atomicrmw_umax_i32:
  case Intrinsic::primate_masked_atomicrmw_umin_i32:
  case Intrinsic::primate_masked_cmpxchg_i32: {
    PointerType *PtrTy = cast<PointerType>(I.getArgOperand(0)->getType());
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    // Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Align(4);
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore |
                 MachineMemOperand::MOVolatile;
    return true;
  }
  }
}

bool PrimateTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                const AddrMode &AM, Type *Ty,
                                                unsigned AS,
                                                Instruction *I) const {
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // Require a 12-bit signed offset.
  if (!isInt<12>(AM.BaseOffs))
    return false;

  switch (AM.Scale) {
  case 0: // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (!AM.HasBaseReg) // allow "r+i".
      break;
    return false; // disallow "r+r" or "r+r+i".
  default:
    return false;
  }

  return true;
}

bool PrimateTargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

bool PrimateTargetLowering::isLegalAddImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

// On PR32, 64-bit integers are split into their high and low parts and held
// in two different registers, so the trunc is free since the low register can
// just be used.
bool PrimateTargetLowering::isTruncateFree(Type *SrcTy, Type *DstTy) const {
  if (Subtarget.is64Bit() || !SrcTy->isIntegerTy() || !DstTy->isIntegerTy())
    return false;
  unsigned SrcBits = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBits = DstTy->getPrimitiveSizeInBits();
  return (SrcBits == 64 && DestBits == 32);
}

bool PrimateTargetLowering::isTruncateFree(EVT SrcVT, EVT DstVT) const {
  if (Subtarget.is64Bit() || SrcVT.isVector() || DstVT.isVector() ||
      !SrcVT.isInteger() || !DstVT.isInteger())
    return false;
  unsigned SrcBits = SrcVT.getSizeInBits();
  unsigned DestBits = DstVT.getSizeInBits();
  return (SrcBits == 64 && DestBits == 32);
}

bool PrimateTargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  // Zexts are free if they can be combined with a load.
  if (auto *LD = dyn_cast<LoadSDNode>(Val)) {
    EVT MemVT = LD->getMemoryVT();
    if ((MemVT == MVT::i8 || MemVT == MVT::i16 ||
         (Subtarget.is64Bit() && MemVT == MVT::i32)) &&
        (LD->getExtensionType() == ISD::NON_EXTLOAD ||
         LD->getExtensionType() == ISD::ZEXTLOAD))
      return true;
  }

  return TargetLowering::isZExtFree(Val, VT2);
}

bool PrimateTargetLowering::isSExtCheaperThanZExt(EVT SrcVT, EVT DstVT) const {
  return Subtarget.is64Bit() && SrcVT == MVT::i32 && DstVT == MVT::i64;
}

bool PrimateTargetLowering::isCheapToSpeculateCttz(Type *Ty) const {
  return Subtarget.hasStdExtZbb();
}

bool PrimateTargetLowering::isCheapToSpeculateCtlz(Type *Ty) const {
  return Subtarget.hasStdExtZbb();
}

bool PrimateTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                       bool ForCodeSize) const {
  if (VT == MVT::f16 && !Subtarget.hasStdExtZfh())
    return false;
  if (VT == MVT::f32 && !Subtarget.hasStdExtF())
    return false;
  if (VT == MVT::f64 && !Subtarget.hasStdExtD())
    return false;
  if (Imm.isNegZero())
    return false;
  return Imm.isZero();
}

MVT PrimateTargetLowering::getRegisterTypeForCallingConv(LLVMContext &Context,
                                                      CallingConv::ID CC,
                                                      EVT VT) const {
  // Use f32 to pass f16 if it is legal and Zfh is not enabled. We might still
  // end up using a GPR but that will be decided based on ABI.
  if (VT == MVT::f16 && Subtarget.hasStdExtF() && !Subtarget.hasStdExtZfh())
    return MVT::f32;

  return TargetLowering::getRegisterTypeForCallingConv(Context, CC, VT);
}

unsigned PrimateTargetLowering::getNumRegistersForCallingConv(LLVMContext &Context,
                                                           CallingConv::ID CC,
                                                           EVT VT) const {
  // Use f32 to pass f16 if it is legal and Zfh is not enabled. We might still
  // end up using a GPR but that will be decided based on ABI.
  if (VT == MVT::f16 && Subtarget.hasStdExtF() && !Subtarget.hasStdExtZfh())
    return 1;

  return TargetLowering::getNumRegistersForCallingConv(Context, CC, VT);
}

// Changes the condition code and swaps operands if necessary, so the SetCC
// operation matches one of the comparisons supported directly by branches
// in the Primate ISA. May adjust compares to favor compare with 0 over compare
// with 1/-1.
static void translateSetCCForBranch(const SDLoc &DL, SDValue &LHS, SDValue &RHS,
                                    ISD::CondCode &CC, SelectionDAG &DAG) {
  // Convert X > -1 to X >= 0.
  if (CC == ISD::SETGT && isAllOnesConstant(RHS)) {
    RHS = DAG.getConstant(0, DL, RHS.getValueType());
    CC = ISD::SETGE;
    return;
  }
  // Convert X < 1 to 0 >= X.
  if (CC == ISD::SETLT && isOneConstant(RHS)) {
    RHS = LHS;
    LHS = DAG.getConstant(0, DL, RHS.getValueType());
    CC = ISD::SETGE;
    return;
  }

  switch (CC) {
  default:
    break;
  case ISD::SETGT:
  case ISD::SETLE:
  case ISD::SETUGT:
  case ISD::SETULE:
    CC = ISD::getSetCCSwappedOperands(CC);
    std::swap(LHS, RHS);
    break;
  }
}

// Return the Primate branch opcode that matches the given DAG integer
// condition code. The CondCode must be one of those supported by the Primate
// ISA (see translateSetCCForBranch).
static unsigned getBranchOpcodeForIntCondCode(ISD::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unsupported CondCode");
  case ISD::SETEQ:
    return Primate::BEQ;
  case ISD::SETNE:
    return Primate::BNE;
  case ISD::SETLT:
    return Primate::BLT;
  case ISD::SETGE:
    return Primate::BGE;
  case ISD::SETULT:
    return Primate::BLTU;
  case ISD::SETUGE:
    return Primate::BGEU;
  }
}

PrimateII::VLMUL PrimateTargetLowering::getLMUL(MVT VT) {
  assert(VT.isScalableVector() && "Expecting a scalable vector type");
  unsigned KnownSize = VT.getSizeInBits().getKnownMinValue();
  if (VT.getVectorElementType() == MVT::i1)
    KnownSize *= 8;

  switch (KnownSize) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case 8:
    return PrimateII::VLMUL::LMUL_F8;
  case 16:
    return PrimateII::VLMUL::LMUL_F4;
  case 32:
    return PrimateII::VLMUL::LMUL_F2;
  case 64:
    return PrimateII::VLMUL::LMUL_1;
  case 128:
    return PrimateII::VLMUL::LMUL_2;
  case 256:
    return PrimateII::VLMUL::LMUL_4;
  case 512:
    return PrimateII::VLMUL::LMUL_8;
  }
}

unsigned PrimateTargetLowering::getRegClassIDForLMUL(PrimateII::VLMUL LMul) {
  switch (LMul) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case PrimateII::VLMUL::LMUL_F8:
  case PrimateII::VLMUL::LMUL_F4:
  case PrimateII::VLMUL::LMUL_F2:
  case PrimateII::VLMUL::LMUL_1:
    return Primate::VRRegClassID;
  case PrimateII::VLMUL::LMUL_2:
    return Primate::VRM2RegClassID;
  case PrimateII::VLMUL::LMUL_4:
    return Primate::VRM4RegClassID;
  case PrimateII::VLMUL::LMUL_8:
    return Primate::VRM8RegClassID;
  }
}

unsigned PrimateTargetLowering::getSubregIndexByMVT(MVT VT, unsigned Index) {
  PrimateII::VLMUL LMUL = getLMUL(VT);
  if (LMUL == PrimateII::VLMUL::LMUL_F8 ||
      LMUL == PrimateII::VLMUL::LMUL_F4 ||
      LMUL == PrimateII::VLMUL::LMUL_F2 ||
      LMUL == PrimateII::VLMUL::LMUL_1) {
    static_assert(Primate::sub_vrm1_7 == Primate::sub_vrm1_0 + 7,
                  "Unexpected subreg numbering");
    return Primate::sub_vrm1_0 + Index;
  }
  if (LMUL == PrimateII::VLMUL::LMUL_2) {
    static_assert(Primate::sub_vrm2_3 == Primate::sub_vrm2_0 + 3,
                  "Unexpected subreg numbering");
    return Primate::sub_vrm2_0 + Index;
  }
  if (LMUL == PrimateII::VLMUL::LMUL_4) {
    static_assert(Primate::sub_vrm4_1 == Primate::sub_vrm4_0 + 1,
                  "Unexpected subreg numbering");
    return Primate::sub_vrm4_0 + Index;
  }
  llvm_unreachable("Invalid vector type.");
}

unsigned PrimateTargetLowering::getRegClassIDForVecVT(MVT VT) {
  if (VT.getVectorElementType() == MVT::i1)
    return Primate::VRRegClassID;
  return getRegClassIDForLMUL(getLMUL(VT));
}

// Attempt to decompose a subvector insert/extract between VecVT and
// SubVecVT via subregister indices. Returns the subregister index that
// can perform the subvector insert/extract with the given element index, as
// well as the index corresponding to any leftover subvectors that must be
// further inserted/extracted within the register class for SubVecVT.
std::pair<unsigned, unsigned>
PrimateTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
    MVT VecVT, MVT SubVecVT, unsigned InsertExtractIdx,
    const PrimateRegisterInfo *TRI) {
  static_assert((Primate::VRM8RegClassID > Primate::VRM4RegClassID &&
                 Primate::VRM4RegClassID > Primate::VRM2RegClassID &&
                 Primate::VRM2RegClassID > Primate::VRRegClassID),
                "Register classes not ordered");
  unsigned VecRegClassID = getRegClassIDForVecVT(VecVT);
  unsigned SubRegClassID = getRegClassIDForVecVT(SubVecVT);
  // Try to compose a subregister index that takes us from the incoming
  // LMUL>1 register class down to the outgoing one. At each step we half
  // the LMUL:
  //   nxv16i32@12 -> nxv2i32: sub_vrm4_1_then_sub_vrm2_1_then_sub_vrm1_0
  // Note that this is not guaranteed to find a subregister index, such as
  // when we are extracting from one VR type to another.
  unsigned SubRegIdx = Primate::NoSubRegister;
  for (const unsigned RCID :
       {Primate::VRM4RegClassID, Primate::VRM2RegClassID, Primate::VRRegClassID})
    if (VecRegClassID > RCID && SubRegClassID <= RCID) {
      VecVT = VecVT.getHalfNumVectorElementsVT();
      bool IsHi =
          InsertExtractIdx >= VecVT.getVectorElementCount().getKnownMinValue();
      SubRegIdx = TRI->composeSubRegIndices(SubRegIdx,
                                            getSubregIndexByMVT(VecVT, IsHi));
      if (IsHi)
        InsertExtractIdx -= VecVT.getVectorElementCount().getKnownMinValue();
    }
  return {SubRegIdx, InsertExtractIdx};
}

// Permit combining of mask vectors as BUILD_VECTOR never expands to scalar
// stores for those types.
bool PrimateTargetLowering::mergeStoresAfterLegalization(EVT VT) const {
  return !Subtarget.usePRVForFixedLengthVectors() ||
         (VT.isFixedLengthVector() && VT.getVectorElementType() == MVT::i1);
}

static bool usePRVForFixedLengthVectorVT(MVT VT,
                                         const PrimateSubtarget &Subtarget) {
  assert(VT.isFixedLengthVector() && "Expected a fixed length vector type!");
  if (!Subtarget.usePRVForFixedLengthVectors())
    return false;

  // We only support a set of vector types with a consistent maximum fixed size
  // across all supported vector element types to avoid legalization issues.
  // Therefore -- since the largest is v1024i8/v512i16/etc -- the largest
  // fixed-length vector type we support is 1024 bytes.
  if (VT.getFixedSizeInBits() > 1024 * 8)
    return false;

  unsigned MinVLen = Subtarget.getMinPRVVectorSizeInBits();

  // Don't use PRV for vectors we cannot scalarize if required.
  switch (VT.getVectorElementType().SimpleTy) {
  // i1 is supported but has different rules.
  default:
    return false;
  case MVT::i1:
    // Masks can only use a single register.
    if (VT.getVectorNumElements() > MinVLen)
      return false;
    MinVLen /= 8;
    break;
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
    break;
  case MVT::f16:
    if (!Subtarget.hasStdExtZfh())
      return false;
    break;
  case MVT::f32:
    if (!Subtarget.hasStdExtF())
      return false;
    break;
  case MVT::f64:
    if (!Subtarget.hasStdExtD())
      return false;
    break;
  }

  unsigned LMul = divideCeil(VT.getSizeInBits(), MinVLen);
  // Don't use PRV for types that don't fit.
  if (LMul > Subtarget.getMaxLMULForFixedLengthVectors())
    return false;

  // TODO: Perhaps an artificial restriction, but worth having whilst getting
  // the base fixed length PRV support in place.
  if (!VT.isPow2VectorType())
    return false;

  return true;
}

bool PrimateTargetLowering::usePRVForFixedLengthVectorVT(MVT VT) const {
  return ::usePRVForFixedLengthVectorVT(VT, Subtarget);
}

// Return the largest legal scalable vector type that matches VT's element type.
static MVT getContainerForFixedLengthVector(const TargetLowering &TLI, MVT VT,
                                            const PrimateSubtarget &Subtarget) {
  // This may be called before legal types are setup.
  assert(((VT.isFixedLengthVector() && TLI.isTypeLegal(VT)) ||
          usePRVForFixedLengthVectorVT(VT, Subtarget)) &&
         "Expected legal fixed length vector!");

  unsigned MinVLen = Subtarget.getMinPRVVectorSizeInBits();

  MVT EltVT = VT.getVectorElementType();
  switch (EltVT.SimpleTy) {
  default:
    llvm_unreachable("unexpected element type for PRV container");
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
  case MVT::f16:
  case MVT::f32:
  case MVT::f64: {
    // We prefer to use LMUL=1 for VLEN sized types. Use fractional lmuls for
    // narrower types, but we can't have a fractional LMUL with demoninator less
    // than 64/SEW.
    unsigned NumElts =
        divideCeil(VT.getVectorNumElements(), MinVLen / Primate::PRVBitsPerBlock);
    return MVT::getScalableVectorVT(EltVT, NumElts);
  }
  }
}

static MVT getContainerForFixedLengthVector(SelectionDAG &DAG, MVT VT,
                                            const PrimateSubtarget &Subtarget) {
  return getContainerForFixedLengthVector(DAG.getTargetLoweringInfo(), VT,
                                          Subtarget);
}

MVT PrimateTargetLowering::getContainerForFixedLengthVector(MVT VT) const {
  return ::getContainerForFixedLengthVector(*this, VT, getSubtarget());
}

// Grow V to consume an entire PRV register.
static SDValue convertToScalableVector(EVT VT, SDValue V, SelectionDAG &DAG,
                                       const PrimateSubtarget &Subtarget) {
  assert(VT.isScalableVector() &&
         "Expected to convert into a scalable vector!");
  assert(V.getValueType().isFixedLengthVector() &&
         "Expected a fixed length vector operand!");
  SDLoc DL(V);
  SDValue Zero = DAG.getConstant(0, DL, Subtarget.getXLenVT());
  return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, DAG.getUNDEF(VT), V, Zero);
}

// Shrink V so it's just big enough to maintain a VT's worth of data.
static SDValue convertFromScalableVector(EVT VT, SDValue V, SelectionDAG &DAG,
                                         const PrimateSubtarget &Subtarget) {
  assert(VT.isFixedLengthVector() &&
         "Expected to convert into a fixed length vector!");
  assert(V.getValueType().isScalableVector() &&
         "Expected a scalable vector operand!");
  SDLoc DL(V);
  SDValue Zero = DAG.getConstant(0, DL, Subtarget.getXLenVT());
  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, V, Zero);
}

// Gets the two common "VL" operands: an all-ones mask and the vector length.
// VecVT is a vector type, either fixed-length or scalable, and ContainerVT is
// the vector type that it is contained in.
static std::pair<SDValue, SDValue>
getDefaultVLOps(MVT VecVT, MVT ContainerVT, SDLoc DL, SelectionDAG &DAG,
                const PrimateSubtarget &Subtarget) {
  llvm_unreachable("Primate has no getDefaultVLOps");
  // assert(ContainerVT.isScalableVector() && "Expecting scalable container type");
  // MVT XLenVT = Subtarget.getXLenVT();
  // SDValue VL = VecVT.isFixedLengthVector()
  //                  ? DAG.getConstant(VecVT.getVectorNumElements(), DL, XLenVT)
  //                  : DAG.getRegister(Primate::X0, XLenVT);
  // MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
  // SDValue Mask = DAG.getNode(PrimateISD::VMSET_VL, DL, MaskVT, VL);
  // return {Mask, VL};
}

// As above but assuming the given type is a scalable vector type.
static std::pair<SDValue, SDValue>
getDefaultScalableVLOps(MVT VecVT, SDLoc DL, SelectionDAG &DAG,
                        const PrimateSubtarget &Subtarget) {
  assert(VecVT.isScalableVector() && "Expecting a scalable vector");
  return getDefaultVLOps(VecVT, VecVT, DL, DAG, Subtarget);
}

// The state of PRV BUILD_VECTOR and VECTOR_SHUFFLE lowering is that very few
// of either is (currently) supported. This can get us into an infinite loop
// where we try to lower a BUILD_VECTOR as a VECTOR_SHUFFLE as a BUILD_VECTOR
// as a ..., etc.
// Until either (or both) of these can reliably lower any node, reporting that
// we don't want to expand BUILD_VECTORs via VECTOR_SHUFFLEs at least breaks
// the infinite loop. Note that this lowers BUILD_VECTOR through the stack,
// which is not desirable.
bool PrimateTargetLowering::shouldExpandBuildVectorWithShuffles(
    EVT VT, unsigned DefinedValues) const {
  return false;
}

bool PrimateTargetLowering::isShuffleMaskLegal(ArrayRef<int> M, EVT VT) const {
  // Only splats are currently supported.
  if (ShuffleVectorSDNode::isSplatMask(M.data(), VT))
    return true;

  return false;
}

static SDValue lowerSPLAT_VECTOR(SDValue Op, SelectionDAG &DAG,
                                 const PrimateSubtarget &Subtarget) {
  llvm_unreachable("Primate should not see a SPLAT_VECTOR");
  // MVT VT = Op.getSimpleValueType();
  // assert(VT.isFixedLengthVector() && "Unexpected vector!");

  // MVT ContainerVT = getContainerForFixedLengthVector(DAG, VT, Subtarget);

  // SDLoc DL(Op);
  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  // unsigned Opc =
  //     VT.isFloatingPoint() ? PrimateISD::VFMV_V_F_VL : PrimateISD::VMV_V_X_VL;
  // SDValue Splat = DAG.getNode(Opc, DL, ContainerVT, Op.getOperand(0), VL);
  // return convertFromScalableVector(VT, Splat, DAG, Subtarget);
}

struct VIDSequence {
  int64_t Step;
  int64_t Addend;
};

// Try to match an arithmetic-sequence BUILD_VECTOR [X,X+S,X+2*S,...,X+(N-1)*S]
// to the (non-zero) step S and start value X. This can be then lowered as the
// PRV sequence (VID * S) + X, for example.
// Note that this method will also match potentially unappealing index
// sequences, like <i32 0, i32 50939494>, however it is left to the caller to
// determine whether this is worth generating code for.
static std::optional<VIDSequence> isSimpleVIDSequence(SDValue Op) {
  unsigned NumElts = Op.getNumOperands();
  assert(Op.getOpcode() == ISD::BUILD_VECTOR && "Unexpected BUILD_VECTOR");
  if (!Op.getValueType().isInteger())
    return {};

  std::optional<int64_t> SeqStep, SeqAddend;
  std::optional<std::pair<uint64_t, unsigned>> PrevElt;
  unsigned EltSizeInBits = Op.getValueType().getScalarSizeInBits();
  for (unsigned Idx = 0; Idx < NumElts; Idx++) {
    // Assume undef elements match the sequence; we just have to be careful
    // when interpolating across them.
    if (Op.getOperand(Idx).isUndef())
      continue;
    // The BUILD_VECTOR must be all constants.
    if (!isa<ConstantSDNode>(Op.getOperand(Idx)))
      return {};

    uint64_t Val = Op.getConstantOperandVal(Idx) &
                   maskTrailingOnes<uint64_t>(EltSizeInBits);

    if (PrevElt) {
      // Calculate the step since the last non-undef element, and ensure
      // it's consistent across the entire sequence.
      int64_t Diff = SignExtend64(Val - PrevElt->first, EltSizeInBits);
      // The difference must cleanly divide the element span.
      if (Diff % (Idx - PrevElt->second) != 0)
        return {};
      int64_t Step = Diff / (Idx - PrevElt->second);
      // A zero step indicates we're either a not an index sequence, or we
      // have a fractional step. This must be handled by a more complex
      // pattern recognition (undefs complicate things here).
      if (Step == 0)
        return {};
      if (!SeqStep)
        SeqStep = Step;
      else if (Step != SeqStep)
        return {};
    }

    // Record and/or check any addend.
    if (SeqStep) {
      int64_t Addend =
          SignExtend64(Val - (Idx * (uint64_t)*SeqStep), EltSizeInBits);
      if (!SeqAddend)
        SeqAddend = Addend;
      else if (SeqAddend != Addend)
        return {};
    }

    // Record this non-undef element for later.
    PrevElt = std::make_pair(Val, Idx);
  }
  // We need to have logged both a step and an addend for this to count as
  // a legal index sequence.
  if (!SeqStep || !SeqAddend)
    return {};

  return VIDSequence{*SeqStep, *SeqAddend};
}

static SDValue lowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG,
                                 const PrimateSubtarget &Subtarget) {
  llvm_unreachable("Primate should not see a BUILD_VECTOR intrinsic");
  // MVT VT = Op.getSimpleValueType();
  // assert(VT.isFixedLengthVector() && "Unexpected vector!");

  // MVT ContainerVT = getContainerForFixedLengthVector(DAG, VT, Subtarget);

  // SDLoc DL(Op);
  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  // MVT XLenVT = Subtarget.getXLenVT();
  // unsigned NumElts = Op.getNumOperands();

  // if (VT.getVectorElementType() == MVT::i1) {
  //   if (ISD::isBuildVectorAllZeros(Op.getNode())) {
  //     SDValue VMClr = DAG.getNode(PrimateISD::VMCLR_VL, DL, ContainerVT, VL);
  //     return convertFromScalableVector(VT, VMClr, DAG, Subtarget);
  //   }

  //   if (ISD::isBuildVectorAllOnes(Op.getNode())) {
  //     SDValue VMSet = DAG.getNode(PrimateISD::VMSET_VL, DL, ContainerVT, VL);
  //     return convertFromScalableVector(VT, VMSet, DAG, Subtarget);
  //   }

  //   // Lower constant mask BUILD_VECTORs via an integer vector type, in
  //   // scalar integer chunks whose bit-width depends on the number of mask
  //   // bits and XLEN.
  //   // First, determine the most appropriate scalar integer type to use. This
  //   // is at most XLenVT, but may be shrunk to a smaller vector element type
  //   // according to the size of the final vector - use i8 chunks rather than
  //   // XLenVT if we're producing a v8i1. This results in more consistent
  //   // codegen across PR32 and PR64.
  //   unsigned NumViaIntegerBits =
  //       std::min(std::max(NumElts, 8u), Subtarget.getXLen());
  //   if (ISD::isBuildVectorOfConstantSDNodes(Op.getNode())) {
  //     // If we have to use more than one INSERT_VECTOR_ELT then this
  //     // optimization is likely to increase code size; avoid peforming it in
  //     // such a case. We can use a load from a constant pool in this case.
  //     if (DAG.shouldOptForSize() && NumElts > NumViaIntegerBits)
  //       return SDValue();
  //     // Now we can create our integer vector type. Note that it may be larger
  //     // than the resulting mask type: v4i1 would use v1i8 as its integer type.
  //     MVT IntegerViaVecVT =
  //         MVT::getVectorVT(MVT::getIntegerVT(NumViaIntegerBits),
  //                          divideCeil(NumElts, NumViaIntegerBits));

  //     uint64_t Bits = 0;
  //     unsigned BitPos = 0, IntegerEltIdx = 0;
  //     SDValue Vec = DAG.getUNDEF(IntegerViaVecVT);

  //     for (unsigned I = 0; I < NumElts; I++, BitPos++) {
  //       // Once we accumulate enough bits to fill our scalar type, insert into
  //       // our vector and clear our accumulated data.
  //       if (I != 0 && I % NumViaIntegerBits == 0) {
  //         if (NumViaIntegerBits <= 32)
  //           Bits = SignExtend64(Bits, 32);
  //         SDValue Elt = DAG.getConstant(Bits, DL, XLenVT);
  //         Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, IntegerViaVecVT, Vec,
  //                           Elt, DAG.getConstant(IntegerEltIdx, DL, XLenVT));
  //         Bits = 0;
  //         BitPos = 0;
  //         IntegerEltIdx++;
  //       }
  //       SDValue V = Op.getOperand(I);
  //       bool BitValue = !V.isUndef() && cast<ConstantSDNode>(V)->getZExtValue();
  //       Bits |= ((uint64_t)BitValue << BitPos);
  //     }

  //     // Insert the (remaining) scalar value into position in our integer
  //     // vector type.
  //     if (NumViaIntegerBits <= 32)
  //       Bits = SignExtend64(Bits, 32);
  //     SDValue Elt = DAG.getConstant(Bits, DL, XLenVT);
  //     Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, IntegerViaVecVT, Vec, Elt,
  //                       DAG.getConstant(IntegerEltIdx, DL, XLenVT));

  //     if (NumElts < NumViaIntegerBits) {
  //       // If we're producing a smaller vector than our minimum legal integer
  //       // type, bitcast to the equivalent (known-legal) mask type, and extract
  //       // our final mask.
  //       assert(IntegerViaVecVT == MVT::v1i8 && "Unexpected mask vector type");
  //       Vec = DAG.getBitcast(MVT::v8i1, Vec);
  //       Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, Vec,
  //                         DAG.getConstant(0, DL, XLenVT));
  //     } else {
  //       // Else we must have produced an integer type with the same size as the
  //       // mask type; bitcast for the final result.
  //       assert(VT.getSizeInBits() == IntegerViaVecVT.getSizeInBits());
  //       Vec = DAG.getBitcast(VT, Vec);
  //     }

  //     return Vec;
  //   }

  //   // A BUILD_VECTOR can be lowered as a SETCC. For each fixed-length mask
  //   // vector type, we have a legal equivalently-sized i8 type, so we can use
  //   // that.
  //   MVT WideVecVT = VT.changeVectorElementType(MVT::i8);
  //   SDValue VecZero = DAG.getConstant(0, DL, WideVecVT);

  //   SDValue WideVec;
  //   if (SDValue Splat = cast<BuildVectorSDNode>(Op)->getSplatValue()) {
  //     // For a splat, perform a scalar truncate before creating the wider
  //     // vector.
  //     assert(Splat.getValueType() == XLenVT &&
  //            "Unexpected type for i1 splat value");
  //     Splat = DAG.getNode(ISD::AND, DL, XLenVT, Splat,
  //                         DAG.getConstant(1, DL, XLenVT));
  //     WideVec = DAG.getSplatBuildVector(WideVecVT, DL, Splat);
  //   } else {
  //     SmallVector<SDValue, 8> Ops(Op->op_values());
  //     WideVec = DAG.getBuildVector(WideVecVT, DL, Ops);
  //     SDValue VecOne = DAG.getConstant(1, DL, WideVecVT);
  //     WideVec = DAG.getNode(ISD::AND, DL, WideVecVT, WideVec, VecOne);
  //   }

  //   return DAG.getSetCC(DL, VT, WideVec, VecZero, ISD::SETNE);
  // }

  // if (SDValue Splat = cast<BuildVectorSDNode>(Op)->getSplatValue()) {
  //   unsigned Opc = VT.isFloatingPoint() ? PrimateISD::VFMV_V_F_VL
  //                                       : PrimateISD::VMV_V_X_VL;
  //   Splat = DAG.getNode(Opc, DL, ContainerVT, Splat, VL);
  //   return convertFromScalableVector(VT, Splat, DAG, Subtarget);
  // }

  // // Try and match index sequences, which we can lower to the vid instruction
  // // with optional modifications. An all-undef vector is matched by
  // // getSplatValue, above.
  // if (auto SimpleVID = isSimpleVIDSequence(Op)) {
  //   int64_t Step = SimpleVID->Step;
  //   int64_t Addend = SimpleVID->Addend;
  //   // Only emit VIDs with suitably-small steps/addends. We use imm5 is a
  //   // threshold since it's the immediate value many PRV instructions accept.
  //   if (isInt<5>(Step) && isInt<5>(Addend)) {
  //     SDValue VID = DAG.getNode(PrimateISD::VID_VL, DL, ContainerVT, Mask, VL);
  //     // Convert right out of the scalable type so we can use standard ISD
  //     // nodes for the rest of the computation. If we used scalable types with
  //     // these, we'd lose the fixed-length vector info and generate worse
  //     // vsetvli code.
  //     VID = convertFromScalableVector(VT, VID, DAG, Subtarget);
  //     assert(Step != 0 && "Invalid step");
  //     bool Negate = false;
  //     if (Step != 1) {
  //       int64_t SplatStepVal = Step;
  //       unsigned Opcode = ISD::MUL;
  //       if (isPowerOf2_64(std::abs(Step))) {
  //         Negate = Step < 0;
  //         Opcode = ISD::SHL;
  //         SplatStepVal = Log2_64(std::abs(Step));
  //       }
  //       SDValue SplatStep = DAG.getSplatVector(
  //           VT, DL, DAG.getConstant(SplatStepVal, DL, XLenVT));
  //       VID = DAG.getNode(Opcode, DL, VT, VID, SplatStep);
  //     }
  //     if (Addend != 0 || Negate) {
  //       SDValue SplatAddend =
  //           DAG.getSplatVector(VT, DL, DAG.getConstant(Addend, DL, XLenVT));
  //       VID = DAG.getNode(Negate ? ISD::SUB : ISD::ADD, DL, VT, SplatAddend, VID);
  //     }
  //     return VID;
  //   }
  // }

  // // Attempt to detect "hidden" splats, which only reveal themselves as splats
  // // when re-interpreted as a vector with a larger element type. For example,
  // //   v4i16 = build_vector i16 0, i16 1, i16 0, i16 1
  // // could be instead splat as
  // //   v2i32 = build_vector i32 0x00010000, i32 0x00010000
  // // TODO: This optimization could also work on non-constant splats, but it
  // // would require bit-manipulation instructions to construct the splat value.
  // SmallVector<SDValue> Sequence;
  // unsigned EltBitSize = VT.getScalarSizeInBits();
  // const auto *BV = cast<BuildVectorSDNode>(Op);
  // if (VT.isInteger() && EltBitSize < 64 &&
  //     ISD::isBuildVectorOfConstantSDNodes(Op.getNode()) &&
  //     BV->getRepeatedSequence(Sequence) &&
  //     (Sequence.size() * EltBitSize) <= 64) {
  //   unsigned SeqLen = Sequence.size();
  //   MVT ViaIntVT = MVT::getIntegerVT(EltBitSize * SeqLen);
  //   MVT ViaVecVT = MVT::getVectorVT(ViaIntVT, NumElts / SeqLen);
  //   assert((ViaIntVT == MVT::i16 || ViaIntVT == MVT::i32 ||
  //           ViaIntVT == MVT::i64) &&
  //          "Unexpected sequence type");

  //   unsigned EltIdx = 0;
  //   uint64_t EltMask = maskTrailingOnes<uint64_t>(EltBitSize);
  //   uint64_t SplatValue = 0;
  //   // Construct the amalgamated value which can be splatted as this larger
  //   // vector type.
  //   for (const auto &SeqV : Sequence) {
  //     if (!SeqV.isUndef())
  //       SplatValue |= ((cast<ConstantSDNode>(SeqV)->getZExtValue() & EltMask)
  //                      << (EltIdx * EltBitSize));
  //     EltIdx++;
  //   }

  //   // On PR64, sign-extend from 32 to 64 bits where possible in order to
  //   // achieve better constant materializion.
  //   if (Subtarget.is64Bit() && ViaIntVT == MVT::i32)
  //     SplatValue = SignExtend64(SplatValue, 32);

  //   // Since we can't introduce illegal i64 types at this stage, we can only
  //   // perform an i64 splat on PR32 if it is its own sign-extended value. That
  //   // way we can use PRV instructions to splat.
  //   assert((ViaIntVT.bitsLE(XLenVT) ||
  //           (!Subtarget.is64Bit() && ViaIntVT == MVT::i64)) &&
  //          "Unexpected bitcast sequence");
  //   if (ViaIntVT.bitsLE(XLenVT) || isInt<32>(SplatValue)) {
  //     SDValue ViaVL =
  //         DAG.getConstant(ViaVecVT.getVectorNumElements(), DL, XLenVT);
  //     MVT ViaContainerVT =
  //         getContainerForFixedLengthVector(DAG, ViaVecVT, Subtarget);
  //     SDValue Splat =
  //         DAG.getNode(PrimateISD::VMV_V_X_VL, DL, ViaContainerVT,
  //                     DAG.getConstant(SplatValue, DL, XLenVT), ViaVL);
  //     Splat = convertFromScalableVector(ViaVecVT, Splat, DAG, Subtarget);
  //     return DAG.getBitcast(VT, Splat);
  //   }
  // }

  // // Try and optimize BUILD_VECTORs with "dominant values" - these are values
  // // which constitute a large proportion of the elements. In such cases we can
  // // splat a vector with the dominant element and make up the shortfall with
  // // INSERT_VECTOR_ELTs.
  // // Note that this includes vectors of 2 elements by association. The
  // // upper-most element is the "dominant" one, allowing us to use a splat to
  // // "insert" the upper element, and an insert of the lower element at position
  // // 0, which improves codegen.
  // SDValue DominantValue;
  // unsigned MostCommonCount = 0;
  // DenseMap<SDValue, unsigned> ValueCounts;
  // unsigned NumUndefElts =
  //     count_if(Op->op_values(), [](const SDValue &V) { return V.isUndef(); });

  // for (SDValue V : Op->op_values()) {
  //   if (V.isUndef())
  //     continue;

  //   ValueCounts.insert(std::make_pair(V, 0));
  //   unsigned &Count = ValueCounts[V];

  //   // Is this value dominant? In case of a tie, prefer the highest element as
  //   // it's cheaper to insert near the beginning of a vector than it is at the
  //   // end.
  //   if (++Count >= MostCommonCount) {
  //     DominantValue = V;
  //     MostCommonCount = Count;
  //   }
  // }

  // assert(DominantValue && "Not expecting an all-undef BUILD_VECTOR");
  // unsigned NumDefElts = NumElts - NumUndefElts;
  // unsigned DominantValueCountThreshold = NumDefElts <= 2 ? 0 : NumDefElts - 2;

  // // Don't perform this optimization when optimizing for size, since
  // // materializing elements and inserting them tends to cause code bloat.
  // if (!DAG.shouldOptForSize() &&
  //     ((MostCommonCount > DominantValueCountThreshold) ||
  //      (ValueCounts.size() <= Log2_32(NumDefElts)))) {
  //   // Start by splatting the most common element.
  //   SDValue Vec = DAG.getSplatBuildVector(VT, DL, DominantValue);

  //   DenseSet<SDValue> Processed{DominantValue};
  //   MVT SelMaskTy = VT.changeVectorElementType(MVT::i1);
  //   for (const auto &OpIdx : enumerate(Op->ops())) {
  //     const SDValue &V = OpIdx.value();
  //     if (V.isUndef() || !Processed.insert(V).second)
  //       continue;
  //     if (ValueCounts[V] == 1) {
  //       Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, VT, Vec, V,
  //                         DAG.getConstant(OpIdx.index(), DL, XLenVT));
  //     } else {
  //       // Blend in all instances of this value using a VSELECT, using a
  //       // mask where each bit signals whether that element is the one
  //       // we're after.
  //       SmallVector<SDValue> Ops;
  //       transform(Op->op_values(), std::back_inserter(Ops), [&](SDValue V1) {
  //         return DAG.getConstant(V == V1, DL, XLenVT);
  //       });
  //       Vec = DAG.getNode(ISD::VSELECT, DL, VT,
  //                         DAG.getBuildVector(SelMaskTy, DL, Ops),
  //                         DAG.getSplatBuildVector(VT, DL, V), Vec);
  //     }
  //   }

  //   return Vec;
  // }

  // return SDValue();
}

static SDValue splatPartsI64WithVL(const SDLoc &DL, MVT VT, SDValue Lo,
                                   SDValue Hi, SDValue VL, SelectionDAG &DAG) {
  llvm_unreachable("Primate shouldn't splat vectors");
  // if (isa<ConstantSDNode>(Lo) && isa<ConstantSDNode>(Hi)) {
  //   int32_t LoC = cast<ConstantSDNode>(Lo)->getSExtValue();
  //   int32_t HiC = cast<ConstantSDNode>(Hi)->getSExtValue();
  //   // If Hi constant is all the same sign bit as Lo, lower this as a custom
  //   // node in order to try and match PRV vector/scalar instructions.
  //   if ((LoC >> 31) == HiC)
  //     return DAG.getNode(PrimateISD::VMV_V_X_VL, DL, VT, Lo, VL);
  // }

  // // Fall back to a stack store and stride x0 vector load.
  // return DAG.getNode(PrimateISD::SPLAT_VECTOR_SPLIT_I64_VL, DL, VT, Lo, Hi, VL);
}

// Called by type legalization to handle splat of i64 on PR32.
// FIXME: We can optimize this when the type has sign or zero bits in one
// of the halves.
static SDValue splatSplitI64WithVL(const SDLoc &DL, MVT VT, SDValue Scalar,
                                   SDValue VL, SelectionDAG &DAG) {
  assert(Scalar.getValueType() == MVT::i64 && "Unexpected VT!");
  SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Scalar,
                           DAG.getConstant(0, DL, MVT::i32));
  SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Scalar,
                           DAG.getConstant(1, DL, MVT::i32));
  return splatPartsI64WithVL(DL, VT, Lo, Hi, VL, DAG);
}

// This function lowers a splat of a scalar operand Splat with the vector
// length VL. It ensures the final sequence is type legal, which is useful when
// lowering a splat after type legalization.
static SDValue lowerScalarSplat(SDValue Scalar, SDValue VL, MVT VT, SDLoc DL,
                                SelectionDAG &DAG,
                                const PrimateSubtarget &Subtarget) {
  llvm_unreachable("Primate should not see a ScalarSplat");
  // if (VT.isFloatingPoint())
  //   return DAG.getNode(PrimateISD::VFMV_V_F_VL, DL, VT, Scalar, VL);

  // MVT XLenVT = Subtarget.getXLenVT();

  // // Simplest case is that the operand needs to be promoted to XLenVT.
  // if (Scalar.getValueType().bitsLE(XLenVT)) {
  //   // If the operand is a constant, sign extend to increase our chances
  //   // of being able to use a .vi instruction. ANY_EXTEND would become a
  //   // a zero extend and the simm5 check in isel would fail.
  //   // FIXME: Should we ignore the upper bits in isel instead?
  //   unsigned ExtOpc =
  //       isa<ConstantSDNode>(Scalar) ? ISD::SIGN_EXTEND : ISD::ANY_EXTEND;
  //   Scalar = DAG.getNode(ExtOpc, DL, XLenVT, Scalar);
  //   return DAG.getNode(PrimateISD::VMV_V_X_VL, DL, VT, Scalar, VL);
  // }

  // assert(XLenVT == MVT::i32 && Scalar.getValueType() == MVT::i64 &&
  //        "Unexpected scalar for splat lowering!");

  // // Otherwise use the more complicated splatting algorithm.
  // return splatSplitI64WithVL(DL, VT, Scalar, VL, DAG);
}

static SDValue lowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG,
                                   const PrimateSubtarget &Subtarget) {
  llvm_unreachable("Primate should not see a vector shuffle");
  // SDValue V1 = Op.getOperand(0);
  // SDValue V2 = Op.getOperand(1);
  // SDLoc DL(Op);
  // MVT XLenVT = Subtarget.getXLenVT();
  // MVT VT = Op.getSimpleValueType();
  // unsigned NumElts = VT.getVectorNumElements();
  // ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op.getNode());

  // MVT ContainerVT = getContainerForFixedLengthVector(DAG, VT, Subtarget);

  // SDValue TrueMask, VL;
  // std::tie(TrueMask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  // if (SVN->isSplat()) {
  //   const int Lane = SVN->getSplatIndex();
  //   if (Lane >= 0) {
  //     MVT SVT = VT.getVectorElementType();

  //     // Turn splatted vector load into a strided load with an X0 stride.
  //     SDValue V = V1;
  //     // Peek through CONCAT_VECTORS as VectorCombine can concat a vector
  //     // with undef.
  //     // FIXME: Peek through INSERT_SUBVECTOR, EXTRACT_SUBVECTOR, bitcasts?
  //     int Offset = Lane;
  //     if (V.getOpcode() == ISD::CONCAT_VECTORS) {
  //       int OpElements =
  //           V.getOperand(0).getSimpleValueType().getVectorNumElements();
  //       V = V.getOperand(Offset / OpElements);
  //       Offset %= OpElements;
  //     }

  //     // We need to ensure the load isn't atomic or volatile.
  //     if (ISD::isNormalLoad(V.getNode()) && cast<LoadSDNode>(V)->isSimple()) {
  //       auto *Ld = cast<LoadSDNode>(V);
  //       Offset *= SVT.getStoreSize();
  //       SDValue NewAddr = DAG.getMemBasePlusOffset(Ld->getBasePtr(),
  //                                                  TypeSize::Fixed(Offset), DL);

  //       // If this is SEW=64 on PR32, use a strided load with a stride of x0.
  //       if (SVT.isInteger() && SVT.bitsGT(XLenVT)) {
  //         SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});
  //         SDValue IntID =
  //             DAG.getTargetConstant(Intrinsic::primate_vlse, DL, XLenVT);
  //         SDValue Ops[] = {Ld->getChain(), IntID, NewAddr,
  //                          DAG.getRegister(Primate::X0, XLenVT), VL};
  //         SDValue NewLoad = DAG.getMemIntrinsicNode(
  //             ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops, SVT,
  //             DAG.getMachineFunction().getMachineMemOperand(
  //                 Ld->getMemOperand(), Offset, SVT.getStoreSize()));
  //         DAG.makeEquivalentMemoryOrdering(Ld, NewLoad);
  //         return convertFromScalableVector(VT, NewLoad, DAG, Subtarget);
  //       }

  //       // Otherwise use a scalar load and splat. This will give the best
  //       // opportunity to fold a splat into the operation. ISel can turn it into
  //       // the x0 strided load if we aren't able to fold away the select.
  //       if (SVT.isFloatingPoint())
  //         V = DAG.getLoad(SVT, DL, Ld->getChain(), NewAddr,
  //                         Ld->getPointerInfo().getWithOffset(Offset),
  //                         Ld->getOriginalAlign(),
  //                         Ld->getMemOperand()->getFlags());
  //       else
  //         V = DAG.getExtLoad(ISD::SEXTLOAD, DL, XLenVT, Ld->getChain(), NewAddr,
  //                            Ld->getPointerInfo().getWithOffset(Offset), SVT,
  //                            Ld->getOriginalAlign(),
  //                            Ld->getMemOperand()->getFlags());
  //       DAG.makeEquivalentMemoryOrdering(Ld, V);

  //       unsigned Opc =
  //           VT.isFloatingPoint() ? PrimateISD::VFMV_V_F_VL : PrimateISD::VMV_V_X_VL;
  //       SDValue Splat = DAG.getNode(Opc, DL, ContainerVT, V, VL);
  //       return convertFromScalableVector(VT, Splat, DAG, Subtarget);
  //     }

  //     V1 = convertToScalableVector(ContainerVT, V1, DAG, Subtarget);
  //     assert(Lane < (int)NumElts && "Unexpected lane!");
  //     SDValue Gather =
  //         DAG.getNode(PrimateISD::VRGATHER_VX_VL, DL, ContainerVT, V1,
  //                     DAG.getConstant(Lane, DL, XLenVT), TrueMask, VL);
  //     return convertFromScalableVector(VT, Gather, DAG, Subtarget);
  //   }
  // }

  // // Detect shuffles which can be re-expressed as vector selects; these are
  // // shuffles in which each element in the destination is taken from an element
  // // at the corresponding index in either source vectors.
  // bool IsSelect = all_of(enumerate(SVN->getMask()), [&](const auto &MaskIdx) {
  //   int MaskIndex = MaskIdx.value();
  //   return MaskIndex < 0 || MaskIdx.index() == (unsigned)MaskIndex % NumElts;
  // });

  // assert(!V1.isUndef() && "Unexpected shuffle canonicalization");

  // SmallVector<SDValue> MaskVals;
  // // As a backup, shuffles can be lowered via a vrgather instruction, possibly
  // // merged with a second vrgather.
  // SmallVector<SDValue> GatherIndicesLHS, GatherIndicesRHS;

  // // By default we preserve the original operand order, and use a mask to
  // // select LHS as true and RHS as false. However, since PRV vector selects may
  // // feature splats but only on the LHS, we may choose to invert our mask and
  // // instead select between RHS and LHS.
  // bool SwapOps = DAG.isSplatValue(V2) && !DAG.isSplatValue(V1);
  // bool InvertMask = IsSelect == SwapOps;

  // // Now construct the mask that will be used by the vselect or blended
  // // vrgather operation. For vrgathers, construct the appropriate indices into
  // // each vector.
  // for (int MaskIndex : SVN->getMask()) {
  //   bool SelectMaskVal = (MaskIndex < (int)NumElts) ^ InvertMask;
  //   MaskVals.push_back(DAG.getConstant(SelectMaskVal, DL, XLenVT));
  //   if (!IsSelect) {
  //     bool IsLHSOrUndefIndex = MaskIndex < (int)NumElts;
  //     GatherIndicesLHS.push_back(IsLHSOrUndefIndex && MaskIndex >= 0
  //                                    ? DAG.getConstant(MaskIndex, DL, XLenVT)
  //                                    : DAG.getUNDEF(XLenVT));
  //     GatherIndicesRHS.push_back(
  //         IsLHSOrUndefIndex ? DAG.getUNDEF(XLenVT)
  //                           : DAG.getConstant(MaskIndex - NumElts, DL, XLenVT));
  //   }
  // }

  // if (SwapOps) {
  //   std::swap(V1, V2);
  //   std::swap(GatherIndicesLHS, GatherIndicesRHS);
  // }

  // assert(MaskVals.size() == NumElts && "Unexpected select-like shuffle");
  // MVT MaskVT = MVT::getVectorVT(MVT::i1, NumElts);
  // SDValue SelectMask = DAG.getBuildVector(MaskVT, DL, MaskVals);

  // if (IsSelect)
  //   return DAG.getNode(ISD::VSELECT, DL, VT, SelectMask, V1, V2);

  // if (VT.getScalarSizeInBits() == 8 && VT.getVectorNumElements() > 256) {
  //   // On such a large vector we're unable to use i8 as the index type.
  //   // FIXME: We could promote the index to i16 and use vrgatherei16, but that
  //   // may involve vector splitting if we're already at LMUL=8, or our
  //   // user-supplied maximum fixed-length LMUL.
  //   return SDValue();
  // }

  // unsigned GatherOpc = PrimateISD::VRGATHER_VV_VL;
  // MVT IndexVT = VT.changeTypeToInteger();
  // // Since we can't introduce illegal index types at this stage, use i16 and
  // // vrgatherei16 if the corresponding index type for plain vrgather is greater
  // // than XLenVT.
  // if (IndexVT.getScalarType().bitsGT(XLenVT)) {
  //   GatherOpc = PrimateISD::VRGATHEREI16_VV_VL;
  //   IndexVT = IndexVT.changeVectorElementType(MVT::i16);
  // }

  // MVT IndexContainerVT =
  //     ContainerVT.changeVectorElementType(IndexVT.getScalarType());

  // SDValue Gather;
  // // TODO: This doesn't trigger for i64 vectors on PR32, since there we
  // // encounter a bitcasted BUILD_VECTOR with low/high i32 values.
  // if (SDValue SplatValue = DAG.getSplatValue(V1, /*LegalTypes*/ true)) {
  //   Gather = lowerScalarSplat(SplatValue, VL, ContainerVT, DL, DAG, Subtarget);
  // } else {
  //   SDValue LHSIndices = DAG.getBuildVector(IndexVT, DL, GatherIndicesLHS);
  //   LHSIndices =
  //       convertToScalableVector(IndexContainerVT, LHSIndices, DAG, Subtarget);

  //   V1 = convertToScalableVector(ContainerVT, V1, DAG, Subtarget);
  //   Gather =
  //       DAG.getNode(GatherOpc, DL, ContainerVT, V1, LHSIndices, TrueMask, VL);
  // }

  // // If a second vector operand is used by this shuffle, blend it in with an
  // // additional vrgather.
  // if (!V2.isUndef()) {
  //   MVT MaskContainerVT = ContainerVT.changeVectorElementType(MVT::i1);
  //   SelectMask =
  //       convertToScalableVector(MaskContainerVT, SelectMask, DAG, Subtarget);

  //   SDValue RHSIndices = DAG.getBuildVector(IndexVT, DL, GatherIndicesRHS);
  //   RHSIndices =
  //       convertToScalableVector(IndexContainerVT, RHSIndices, DAG, Subtarget);

  //   V2 = convertToScalableVector(ContainerVT, V2, DAG, Subtarget);
  //   V2 = DAG.getNode(GatherOpc, DL, ContainerVT, V2, RHSIndices, TrueMask, VL);
  //   Gather = DAG.getNode(PrimateISD::VSELECT_VL, DL, ContainerVT, SelectMask, V2,
  //                        Gather, VL);
  // }

  // return convertFromScalableVector(VT, Gather, DAG, Subtarget);
}

static SDValue getPRVFPExtendOrRound(SDValue Op, MVT VT, MVT ContainerVT,
                                     SDLoc DL, SelectionDAG &DAG,
                                     const PrimateSubtarget &Subtarget) {
  llvm_unreachable("Primate should not see a FP Vector Extend");
  // if (VT.isScalableVector())
  //   return DAG.getFPExtendOrRound(Op, DL, VT);
  // assert(VT.isFixedLengthVector() &&
  //        "Unexpected value type for PRV FP extend/round lowering");
  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);
  // unsigned PRVOpc = ContainerVT.bitsGT(Op.getSimpleValueType())
  //                       ? PrimateISD::FP_EXTEND_VL
  //                       : PrimateISD::FP_ROUND_VL;
  // return DAG.getNode(PRVOpc, DL, ContainerVT, Op, Mask, VL);
}

// While PRV has alignment restrictions, we should always be able to load as a
// legal equivalently-sized byte-typed vector instead. This method is
// responsible for re-expressing a ISD::LOAD via a correctly-aligned type. If
// the load is already correctly-aligned, it returns SDValue().
SDValue PrimateTargetLowering::expandUnalignedPRVLoad(SDValue Op,
                                                    SelectionDAG &DAG) const {
  auto *Load = cast<LoadSDNode>(Op);
  assert(Load && Load->getMemoryVT().isVector() && "Expected vector load");

  if (allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                     Load->getMemoryVT(),
                                     *Load->getMemOperand()))
    return SDValue();

  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  unsigned EltSizeBits = VT.getScalarSizeInBits();
  assert((EltSizeBits == 16 || EltSizeBits == 32 || EltSizeBits == 64) &&
         "Unexpected unaligned PRV load type");
  MVT NewVT =
      MVT::getVectorVT(MVT::i8, VT.getVectorElementCount() * (EltSizeBits / 8));
  assert(NewVT.isValid() &&
         "Expecting equally-sized PRV vector types to be legal");
  SDValue L = DAG.getLoad(NewVT, DL, Load->getChain(), Load->getBasePtr(),
                          Load->getPointerInfo(), Load->getOriginalAlign(),
                          Load->getMemOperand()->getFlags());
  return DAG.getMergeValues({DAG.getBitcast(VT, L), L.getValue(1)}, DL);
}

// While PRV has alignment restrictions, we should always be able to store as a
// legal equivalently-sized byte-typed vector instead. This method is
// responsible for re-expressing a ISD::STORE via a correctly-aligned type. It
// returns SDValue() if the store is already correctly aligned.
SDValue PrimateTargetLowering::expandUnalignedPRVStore(SDValue Op,
                                                     SelectionDAG &DAG) const {
  auto *Store = cast<StoreSDNode>(Op);
  assert(Store && Store->getValue().getValueType().isVector() &&
         "Expected vector store");

  if (allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                     Store->getMemoryVT(),
                                     *Store->getMemOperand()))
    return SDValue();

  SDLoc DL(Op);
  SDValue StoredVal = Store->getValue();
  MVT VT = StoredVal.getSimpleValueType();
  unsigned EltSizeBits = VT.getScalarSizeInBits();
  assert((EltSizeBits == 16 || EltSizeBits == 32 || EltSizeBits == 64) &&
         "Unexpected unaligned PRV store type");
  MVT NewVT =
      MVT::getVectorVT(MVT::i8, VT.getVectorElementCount() * (EltSizeBits / 8));
  assert(NewVT.isValid() &&
         "Expecting equally-sized PRV vector types to be legal");
  StoredVal = DAG.getBitcast(NewVT, StoredVal);
  return DAG.getStore(Store->getChain(), DL, StoredVal, Store->getBasePtr(),
                      Store->getPointerInfo(), Store->getOriginalAlign(),
                      Store->getMemOperand()->getFlags());
}

SDValue PrimateTargetLowering::LowerOperation(SDValue Op,
                                            SelectionDAG &DAG) const {
  dbgs() << "kayvan lowerOperation()\n";
    switch (Op.getOpcode()) {
  default:
    report_fatal_error("unimplemented operand");
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::BlockAddress:
    return lowerBlockAddress(Op, DAG);
  case ISD::ConstantPool:
    return lowerConstantPool(Op, DAG);
  case ISD::JumpTable:
    return lowerJumpTable(Op, DAG);
  case ISD::GlobalTLSAddress:
    return lowerGlobalTLSAddress(Op, DAG);
  case ISD::SELECT:
    return lowerSELECT(Op, DAG);
  case ISD::BRCOND:
    return lowerBRCOND(Op, DAG);
  case ISD::VASTART:
    return lowerVASTART(Op, DAG);
  case ISD::FRAMEADDR:
    return lowerFRAMEADDR(Op, DAG);
  case ISD::RETURNADDR:
    return lowerRETURNADDR(Op, DAG);
  case ISD::SHL_PARTS:
    return lowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return lowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return lowerShiftRightParts(Op, DAG, false);
  case ISD::BITCAST: {
    SDLoc DL(Op);
    EVT VT = Op.getValueType();
    SDValue Op0 = Op.getOperand(0);
    EVT Op0VT = Op0.getValueType();
    MVT XLenVT = Subtarget.getXLenVT();
    if (VT.isFixedLengthVector()) {
      // We can handle fixed length vector bitcasts with a simple replacement
      // in isel.
      if (Op0VT.isFixedLengthVector())
        return Op;
      // When bitcasting from scalar to fixed-length vector, insert the scalar
      // into a one-element vector of the result type, and perform a vector
      // bitcast.
      if (!Op0VT.isVector()) {
        auto BVT = EVT::getVectorVT(*DAG.getContext(), Op0VT, 1);
        return DAG.getBitcast(VT, DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, BVT,
                                              DAG.getUNDEF(BVT), Op0,
                                              DAG.getConstant(0, DL, XLenVT)));
      }
      return SDValue();
    }
    // Custom-legalize bitcasts from fixed-length vector types to scalar types
    // thus: bitcast the vector to a one-element vector type whose element type
    // is the same as the result type, and extract the first element.
    if (!VT.isVector() && Op0VT.isFixedLengthVector()) {
      LLVMContext &Context = *DAG.getContext();
      SDValue BVec = DAG.getBitcast(EVT::getVectorVT(Context, VT, 1), Op0);
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, BVec,
                         DAG.getConstant(0, DL, XLenVT));
    }
    if (VT == MVT::f16 && Op0VT == MVT::i16 && Subtarget.hasStdExtZfh()) {
      SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, XLenVT, Op0);
      SDValue FPConv = DAG.getNode(PrimateISD::FMV_H_X, DL, MVT::f16, NewOp0);
      return FPConv;
    }
    if (VT == MVT::f32 && Op0VT == MVT::i32 && Subtarget.is64Bit() &&
        Subtarget.hasStdExtF()) {
      SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op0);
      SDValue FPConv =
          DAG.getNode(PrimateISD::FMV_W_X_PR64, DL, MVT::f32, NewOp0);
      return FPConv;
    }
    return SDValue();
  }
  case ISD::INTRINSIC_WO_CHAIN:
    return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN:
    return LowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::BSWAP:
  case ISD::BITREVERSE: {
    // Convert BSWAP/BITREVERSE to GREVI to enable GREVI combinining.
    assert(Subtarget.hasStdExtZbb() && "Unexpected custom legalisation");
    MVT VT = Op.getSimpleValueType();
    SDLoc DL(Op);
    // Start with the maximum immediate value which is the bitwidth - 1.
    unsigned Imm = VT.getSizeInBits() - 1;
    // If this is BSWAP rather than BITREVERSE, clear the lower 3 bits.
    if (Op.getOpcode() == ISD::BSWAP)
      Imm &= ~0x7U;
    return DAG.getNode(PrimateISD::GREV, DL, VT, Op.getOperand(0),
                       DAG.getConstant(Imm, DL, VT));
  }
  case ISD::FSHL:
  case ISD::FSHR: {
    MVT VT = Op.getSimpleValueType();
    assert(VT == Subtarget.getXLenVT() && "Unexpected custom legalization");
    SDLoc DL(Op);
    if (Op.getOperand(2).getOpcode() == ISD::Constant)
      return Op;
    // FSL/FSR take a log2(XLen)+1 bit shift amount but XLenVT FSHL/FSHR only
    // use log(XLen) bits. Mask the shift amount accordingly.
    unsigned ShAmtWidth = Subtarget.getXLen() - 1;
    SDValue ShAmt = DAG.getNode(ISD::AND, DL, VT, Op.getOperand(2),
                                DAG.getConstant(ShAmtWidth, DL, VT));
    unsigned Opc = Op.getOpcode() == ISD::FSHL ? PrimateISD::FSL : PrimateISD::FSR;
    return DAG.getNode(Opc, DL, VT, Op.getOperand(0), Op.getOperand(1), ShAmt);
  }
  case ISD::TRUNCATE: {
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    // Only custom-lower vector truncates
    if (!VT.isVector())
      return Op;

    llvm_unreachable("Primate should not TRUNACATE Vector types");
    // Truncates to mask types are handled differently
    if (VT.getVectorElementType() == MVT::i1)
      return lowerVectorMaskTrunc(Op, DAG);

    // PRV only has truncates which operate from SEW*2->SEW, so lower arbitrary
    // truncates as a series of "PrimateISD::TRUNCATE_VECTOR_VL" nodes which
    // truncate by one power of two at a time.
    MVT DstEltVT = VT.getVectorElementType();

    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();
    MVT SrcEltVT = SrcVT.getVectorElementType();

    assert(DstEltVT.bitsLT(SrcEltVT) &&
           isPowerOf2_64(DstEltVT.getSizeInBits()) &&
           isPowerOf2_64(SrcEltVT.getSizeInBits()) &&
           "Unexpected vector truncate lowering");

    MVT ContainerVT = SrcVT;
    if (SrcVT.isFixedLengthVector()) {
      ContainerVT = getContainerForFixedLengthVector(SrcVT);
      Src = convertToScalableVector(ContainerVT, Src, DAG, Subtarget);
    }

    SDValue Result = Src;
    SDValue Mask, VL;
    std::tie(Mask, VL) =
        getDefaultVLOps(SrcVT, ContainerVT, DL, DAG, Subtarget);
    LLVMContext &Context = *DAG.getContext();
    const ElementCount Count = ContainerVT.getVectorElementCount();
    do {
      SrcEltVT = MVT::getIntegerVT(SrcEltVT.getSizeInBits() / 2);
      EVT ResultVT = EVT::getVectorVT(Context, SrcEltVT, Count);
      //Result = DAG.getNode(PrimateISD::TRUNCATE_VECTOR_VL, DL, ResultVT, Result,
      //                     Mask, VL);
    } while (SrcEltVT != DstEltVT);

    if (SrcVT.isFixedLengthVector())
      Result = convertFromScalableVector(VT, Result, DAG, Subtarget);

    return Result;
  }
  case ISD::ANY_EXTEND:
  case ISD::ZERO_EXTEND:
    llvm_unreachable("Primate should not custom lower ZEXT");
    // if (Op.getOperand(0).getValueType().isVector() &&
    //     Op.getOperand(0).getValueType().getVectorElementType() == MVT::i1)
    //   return lowerVectorMaskExt(Op, DAG, /*ExtVal*/ 1);
    // return lowerFixedLengthVectorExtendToPRV(Op, DAG, PrimateISD::VZEXT_VL);
  case ISD::SIGN_EXTEND:
    llvm_unreachable("Primate should not custom lower SEXT");
    // if (Op.getOperand(0).getValueType().isVector() &&
    //     Op.getOperand(0).getValueType().getVectorElementType() == MVT::i1)
    //   return lowerVectorMaskExt(Op, DAG, /*ExtVal*/ -1);
    // return lowerFixedLengthVectorExtendToPRV(Op, DAG, PrimateISD::VSEXT_VL);
  case ISD::SPLAT_VECTOR_PARTS:
    return lowerSPLAT_VECTOR_PARTS(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    return lowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return lowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::VSCALE: {
    llvm_unreachable("Primate should not see VSCALE");
    // MVT VT = Op.getSimpleValueType();
    // SDLoc DL(Op);
    // SDValue VLENB = DAG.getNode(PrimateISD::READ_VLENB, DL, VT);
    // // We define our scalable vector types for lmul=1 to use a 64 bit known
    // // minimum size. e.g. <vscale x 2 x i32>. VLENB is in bytes so we calculate
    // // vscale as VLENB / 8.
    // assert(Primate::PRVBitsPerBlock == 64 && "Unexpected bits per block!");
    // if (isa<ConstantSDNode>(Op.getOperand(0))) {
    //   // We assume VLENB is a multiple of 8. We manually choose the best shift
    //   // here because SimplifyDemandedBits isn't always able to simplify it.
    //   uint64_t Val = Op.getConstantOperandVal(0);
    //   if (isPowerOf2_64(Val)) {
    //     uint64_t Log2 = Log2_64(Val);
    //     if (Log2 < 3)
    //       return DAG.getNode(ISD::SRL, DL, VT, VLENB,
    //                          DAG.getConstant(3 - Log2, DL, VT));
    //     if (Log2 > 3)
    //       return DAG.getNode(ISD::SHL, DL, VT, VLENB,
    //                          DAG.getConstant(Log2 - 3, DL, VT));
    //     return VLENB;
    //   }
    //   // If the multiplier is a multiple of 8, scale it down to avoid needing
    //   // to shift the VLENB value.
    //   if ((Val % 8) == 0)
    //     return DAG.getNode(ISD::MUL, DL, VT, VLENB,
    //                        DAG.getConstant(Val / 8, DL, VT));
    // }

    // SDValue VScale = DAG.getNode(ISD::SRL, DL, VT, VLENB,
    //                              DAG.getConstant(3, DL, VT));
    // return DAG.getNode(ISD::MUL, DL, VT, VScale, Op.getOperand(0));
  }
  case ISD::FP_EXTEND: {
    // PRV can only do fp_extend to types double the size as the source. We
    // custom-lower f16->f64 extensions to two hops of ISD::FP_EXTEND, going
    // via f32.
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();

    // Prepare any fixed-length vector operands.
    MVT ContainerVT = VT;
    if (SrcVT.isFixedLengthVector()) {
      ContainerVT = getContainerForFixedLengthVector(VT);
      MVT SrcContainerVT =
          ContainerVT.changeVectorElementType(SrcVT.getVectorElementType());
      Src = convertToScalableVector(SrcContainerVT, Src, DAG, Subtarget);
    }

    if (!VT.isVector() || VT.getVectorElementType() != MVT::f64 ||
        SrcVT.getVectorElementType() != MVT::f16) {
      // For scalable vectors, we only need to close the gap between
      // vXf16->vXf64.
      if (!VT.isFixedLengthVector())
        return Op;
      // For fixed-length vectors, lower the FP_EXTEND to a custom "VL" version.
      Src = getPRVFPExtendOrRound(Src, VT, ContainerVT, DL, DAG, Subtarget);
      return convertFromScalableVector(VT, Src, DAG, Subtarget);
    }

    MVT InterVT = VT.changeVectorElementType(MVT::f32);
    MVT InterContainerVT = ContainerVT.changeVectorElementType(MVT::f32);
    SDValue IntermediateExtend = getPRVFPExtendOrRound(
        Src, InterVT, InterContainerVT, DL, DAG, Subtarget);

    SDValue Extend = getPRVFPExtendOrRound(IntermediateExtend, VT, ContainerVT,
                                           DL, DAG, Subtarget);
    if (VT.isFixedLengthVector())
      return convertFromScalableVector(VT, Extend, DAG, Subtarget);
    return Extend;
  }
  case ISD::FP_ROUND: {
    // PRV can only do fp_round to types half the size as the source. We
    // custom-lower f64->f16 rounds via PRV's round-to-odd float
    // conversion instruction.
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();

    // Prepare any fixed-length vector operands.
    MVT ContainerVT = VT;
    if (VT.isFixedLengthVector()) {
      MVT SrcContainerVT = getContainerForFixedLengthVector(SrcVT);
      ContainerVT =
          SrcContainerVT.changeVectorElementType(VT.getVectorElementType());
      Src = convertToScalableVector(SrcContainerVT, Src, DAG, Subtarget);
    }

    if (!VT.isVector() || VT.getVectorElementType() != MVT::f16 ||
        SrcVT.getVectorElementType() != MVT::f64) {
      // For scalable vectors, we only need to close the gap between
      // vXf64<->vXf16.
      if (!VT.isFixedLengthVector())
        return Op;
      // For fixed-length vectors, lower the FP_ROUND to a custom "VL" version.
      Src = getPRVFPExtendOrRound(Src, VT, ContainerVT, DL, DAG, Subtarget);
      return convertFromScalableVector(VT, Src, DAG, Subtarget);
    }
    llvm_unreachable("Primate should not see a vector for FP Rounding");

    // SDValue Mask, VL;
    // std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

    // MVT InterVT = ContainerVT.changeVectorElementType(MVT::f32);
    // SDValue IntermediateRound =
    //     DAG.getNode(PrimateISD::VFNCVT_ROD_VL, DL, InterVT, Src, Mask, VL);
    // SDValue Round = getPRVFPExtendOrRound(IntermediateRound, VT, ContainerVT,
    //                                       DL, DAG, Subtarget);

    // if (VT.isFixedLengthVector())
    //   return convertFromScalableVector(VT, Round, DAG, Subtarget);
    // return Round;
  }
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: {
    // PRV can only do fp<->int conversions to types half/double the size as
    // the source. We custom-lower any conversions that do two hops into
    // sequences.
    MVT VT = Op.getSimpleValueType();
    if (!VT.isVector())
      return Op;
    llvm_unreachable("Primate should not see a vector for type conversions to FP");
    // SDLoc DL(Op);
    // SDValue Src = Op.getOperand(0);
    // MVT EltVT = VT.getVectorElementType();
    // MVT SrcVT = Src.getSimpleValueType();
    // MVT SrcEltVT = SrcVT.getVectorElementType();
    // unsigned EltSize = EltVT.getSizeInBits();
    // unsigned SrcEltSize = SrcEltVT.getSizeInBits();
    // assert(isPowerOf2_32(EltSize) && isPowerOf2_32(SrcEltSize) &&
    //        "Unexpected vector element types");

    // bool IsInt2FP = SrcEltVT.isInteger();
    // // Widening conversions
    // if (EltSize > SrcEltSize && (EltSize / SrcEltSize >= 4)) {
    //   if (IsInt2FP) {
    //     // Do a regular integer sign/zero extension then convert to float.
    //     MVT IVecVT = MVT::getVectorVT(MVT::getIntegerVT(EltVT.getSizeInBits()),
    //                                   VT.getVectorElementCount());
    //     unsigned ExtOpcode = Op.getOpcode() == ISD::UINT_TO_FP
    //                              ? ISD::ZERO_EXTEND
    //                              : ISD::SIGN_EXTEND;
    //     SDValue Ext = DAG.getNode(ExtOpcode, DL, IVecVT, Src);
    //     return DAG.getNode(Op.getOpcode(), DL, VT, Ext);
    //   }
    //   // FP2Int
    //   assert(SrcEltVT == MVT::f16 && "Unexpected FP_TO_[US]INT lowering");
    //   // Do one doubling fp_extend then complete the operation by converting
    //   // to int.
    //   MVT InterimFVT = MVT::getVectorVT(MVT::f32, VT.getVectorElementCount());
    //   SDValue FExt = DAG.getFPExtendOrRound(Src, DL, InterimFVT);
    //   return DAG.getNode(Op.getOpcode(), DL, VT, FExt);
    // }

    // // Narrowing conversions
    // if (SrcEltSize > EltSize && (SrcEltSize / EltSize >= 4)) {
    //   if (IsInt2FP) {
    //     // One narrowing int_to_fp, then an fp_round.
    //     assert(EltVT == MVT::f16 && "Unexpected [US]_TO_FP lowering");
    //     MVT InterimFVT = MVT::getVectorVT(MVT::f32, VT.getVectorElementCount());
    //     SDValue Int2FP = DAG.getNode(Op.getOpcode(), DL, InterimFVT, Src);
    //     return DAG.getFPExtendOrRound(Int2FP, DL, VT);
    //   }
    //   // FP2Int
    //   // One narrowing fp_to_int, then truncate the integer. If the float isn't
    //   // representable by the integer, the result is poison.
    //   MVT IVecVT =
    //       MVT::getVectorVT(MVT::getIntegerVT(SrcEltVT.getSizeInBits() / 2),
    //                        VT.getVectorElementCount());
    //   SDValue FP2Int = DAG.getNode(Op.getOpcode(), DL, IVecVT, Src);
    //   return DAG.getNode(ISD::TRUNCATE, DL, VT, FP2Int);
    // }

    // // Scalable vectors can exit here. Patterns will handle equally-sized
    // // conversions halving/doubling ones.
    // if (!VT.isFixedLengthVector())
    //   return Op;

    // // For fixed-length vectors we lower to a custom "VL" node.
    // unsigned PRVOpc = 0;
    // switch (Op.getOpcode()) {
    // default:
    //   llvm_unreachable("Impossible opcode");
    // case ISD::FP_TO_SINT:
    //   PRVOpc = PrimateISD::FP_TO_SINT_VL;
    //   break;
    // case ISD::FP_TO_UINT:
    //   PRVOpc = PrimateISD::FP_TO_UINT_VL;
    //   break;
    // case ISD::SINT_TO_FP:
    //   PRVOpc = PrimateISD::SINT_TO_FP_VL;
    //   break;
    // case ISD::UINT_TO_FP:
    //   PRVOpc = PrimateISD::UINT_TO_FP_VL;
    //   break;
    // }

    // MVT ContainerVT, SrcContainerVT;
    // // Derive the reference container type from the larger vector type.
    // if (SrcEltSize > EltSize) {
    //   SrcContainerVT = getContainerForFixedLengthVector(SrcVT);
    //   ContainerVT =
    //       SrcContainerVT.changeVectorElementType(VT.getVectorElementType());
    // } else {
    //   ContainerVT = getContainerForFixedLengthVector(VT);
    //   SrcContainerVT = ContainerVT.changeVectorElementType(SrcEltVT);
    // }

    // SDValue Mask, VL;
    // std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

    // Src = convertToScalableVector(SrcContainerVT, Src, DAG, Subtarget);
    // Src = DAG.getNode(PRVOpc, DL, ContainerVT, Src, Mask, VL);
    // return convertFromScalableVector(VT, Src, DAG, Subtarget);
  }
  case ISD::VECREDUCE_ADD:
  case ISD::VECREDUCE_UMAX:
  case ISD::VECREDUCE_SMAX:
  case ISD::VECREDUCE_UMIN:
  case ISD::VECREDUCE_SMIN:
    return lowerVECREDUCE(Op, DAG);
  case ISD::VECREDUCE_AND:
  case ISD::VECREDUCE_OR:
  case ISD::VECREDUCE_XOR:
    if (Op.getOperand(0).getValueType().getVectorElementType() == MVT::i1)
      return lowerVectorMaskVECREDUCE(Op, DAG);
    return lowerVECREDUCE(Op, DAG);
  case ISD::VECREDUCE_FADD:
  case ISD::VECREDUCE_SEQ_FADD:
  case ISD::VECREDUCE_FMIN:
  case ISD::VECREDUCE_FMAX:
    return lowerFPVECREDUCE(Op, DAG);
  case ISD::INSERT_SUBVECTOR:
    return lowerINSERT_SUBVECTOR(Op, DAG);
  case ISD::EXTRACT_SUBVECTOR:
    return lowerEXTRACT_SUBVECTOR(Op, DAG);
  case ISD::STEP_VECTOR:
    return lowerSTEP_VECTOR(Op, DAG);
  case ISD::VECTOR_REVERSE:
    return lowerVECTOR_REVERSE(Op, DAG);
  case ISD::BUILD_VECTOR:
    return lowerBUILD_VECTOR(Op, DAG, Subtarget);
  case ISD::SPLAT_VECTOR:
    if (Op.getValueType().getVectorElementType() == MVT::i1)
      return lowerVectorMaskSplat(Op, DAG);
    return lowerSPLAT_VECTOR(Op, DAG, Subtarget);
  case ISD::VECTOR_SHUFFLE:
    return lowerVECTOR_SHUFFLE(Op, DAG, Subtarget);
  case ISD::CONCAT_VECTORS: {
    // Split CONCAT_VECTORS into a series of INSERT_SUBVECTOR nodes. This is
    // better than going through the stack, as the default expansion does.
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    unsigned NumOpElts =
        Op.getOperand(0).getSimpleValueType().getVectorMinNumElements();
    SDValue Vec = DAG.getUNDEF(VT);
    for (const auto &OpIdx : enumerate(Op->ops()))
      Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, Vec, OpIdx.value(),
                        DAG.getIntPtrConstant(OpIdx.index() * NumOpElts, DL));
    return Vec;
  }
  case ISD::LOAD:
    if (auto V = expandUnalignedPRVLoad(Op, DAG))
      return V;
    if (Op.getValueType().isFixedLengthVector())
      return lowerFixedLengthVectorLoadToPRV(Op, DAG);
    return Op;
  case ISD::STORE:
    if (auto V = expandUnalignedPRVStore(Op, DAG))
      return V;
    if (Op.getOperand(1).getValueType().isFixedLengthVector())
      return lowerFixedLengthVectorStoreToPRV(Op, DAG);
    return Op;
  case ISD::MLOAD:
    return lowerMLOAD(Op, DAG);
  case ISD::MSTORE:
    return lowerMSTORE(Op, DAG);
  case ISD::SETCC:
    return lowerFixedLengthVectorSetccToPRV(Op, DAG);
  case ISD::ADD:
    llvm_unreachable("Primate should not custom lower ADDs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::ADD_VL);
  case ISD::SUB:
    llvm_unreachable("Primate should not custom lower SUBs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::SUB_VL);
  case ISD::MUL:
    llvm_unreachable("Primate should not custom lower MULs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::MUL_VL);
  case ISD::MULHS:
    llvm_unreachable("Primate should not custom lower MULHSs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::MULHS_VL);
  case ISD::MULHU:
    llvm_unreachable("Primate should not custom lower MULHUs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::MULHU_VL);
  case ISD::AND:
    llvm_unreachable("Primate should not custom lower ANDs (Vector)");
    // return lowerFixedLengthVectorLogicOpToPRV(Op, DAG, PrimateISD::VMAND_VL,
                                              // PrimateISD::AND_VL);
  case ISD::OR:
    llvm_unreachable("Primate should not custom lower ORs (Vector)");
    // return lowerFixedLengthVectorLogicOpToPRV(Op, DAG, PrimateISD::VMOR_VL,
                                              // PrimateISD::OR_VL);
  case ISD::XOR:
    llvm_unreachable("Primate should not custom lower XORs (Vector)");
    // return lowerFixedLengthVectorLogicOpToPRV(Op, DAG, PrimateISD::VMXOR_VL,
                                              // PrimateISD::XOR_VL);
  case ISD::SDIV:
    llvm_unreachable("Primate should not custom lower SDIVs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::SDIV_VL);
  case ISD::SREM:
    llvm_unreachable("Primate should not custom lower SREMs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::SREM_VL);
  case ISD::UDIV:
    llvm_unreachable("Primate should not custom lower UDIVs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::UDIV_VL);
  case ISD::UREM:
    llvm_unreachable("Primate should not custom lower UREMs (Vector)");
    // return lowerToScalableOp(Op, DAG, PrimateISD::UREM_VL);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    if (Op.getSimpleValueType().isFixedLengthVector())
      return lowerFixedLengthVectorShiftToPRV(Op, DAG);
    // This can be called for an i32 shift amount that needs to be promoted.
    assert(Op.getOperand(1).getValueType() == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    return SDValue();
  case ISD::SADDSAT:
  case ISD::UADDSAT:
  case ISD::SSUBSAT:
  case ISD::USUBSAT:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FMA:
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
    llvm_unreachable("Primate Vector custom lower bad");
  case ISD::ABS:
    return lowerABS(Op, DAG);
  case ISD::VSELECT:
    return lowerFixedLengthVectorSelectToPRV(Op, DAG);
  case ISD::FCOPYSIGN:
    return lowerFixedLengthVectorFCOPYSIGNToPRV(Op, DAG);
  case ISD::MGATHER:
    return lowerMGATHER(Op, DAG);
  case ISD::MSCATTER:
    return lowerMSCATTER(Op, DAG);
  case ISD::SET_ROUNDING:
    return lowerSET_ROUNDING(Op, DAG);
  case ISD::VP_ADD:
  case ISD::VP_SUB:
  case ISD::VP_MUL:
  case ISD::VP_SDIV:
  case ISD::VP_UDIV:
  case ISD::VP_SREM:
  case ISD::VP_UREM:
  case ISD::VP_AND:
  case ISD::VP_OR:
  case ISD::VP_XOR:
  case ISD::VP_ASHR:
  case ISD::VP_LSHR:
  case ISD::VP_SHL:
  case ISD::VP_FADD:
  case ISD::VP_FSUB:
  case ISD::VP_FMUL:
  case ISD::VP_FDIV:
    llvm_unreachable("Primate custom lower Vector thing");
  }
}

static SDValue getTargetNode(GlobalAddressSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, Flags);
}

static SDValue getTargetNode(BlockAddressSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetBlockAddress(N->getBlockAddress(), Ty, N->getOffset(),
                                   Flags);
}

static SDValue getTargetNode(ConstantPoolSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetConstantPool(N->getConstVal(), Ty, N->getAlign(),
                                   N->getOffset(), Flags);
}

static SDValue getTargetNode(JumpTableSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetJumpTable(N->getIndex(), Ty, Flags);
}

template <class NodeTy>
SDValue PrimateTargetLowering::getAddr(NodeTy *N, SelectionDAG &DAG,
                                     bool IsLocal) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());

  if (isPositionIndependent()) {
    SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);
    if (IsLocal)
      // Use PC-relative addressing to access the symbol. This generates the
      // pattern (PseudoLLA sym), which expands to (addi (auipc %pcrel_hi(sym))
      // %pcrel_lo(auipc)).
      return SDValue(DAG.getMachineNode(Primate::PseudoLLA, DL, Ty, Addr), 0);

    // Use PC-relative addressing to access the GOT for this symbol, then load
    // the address from the GOT. This generates the pattern (PseudoLA sym),
    // which expands to (ld (addi (auipc %got_pcrel_hi(sym)) %pcrel_lo(auipc))).
    return SDValue(DAG.getMachineNode(Primate::PseudoLA, DL, Ty, Addr), 0);
  }

  switch (getTargetMachine().getCodeModel()) {
  default:
    report_fatal_error("Unsupported code model for lowering");
  case CodeModel::Small: {
    // Generate a sequence for accessing addresses within the first 2 GiB of
    // address space. This generates the pattern (addi (lui %hi(sym)) %lo(sym)).
    SDValue AddrHi = getTargetNode(N, DL, Ty, DAG, PrimateII::MO_HI);
    SDValue AddrLo = getTargetNode(N, DL, Ty, DAG, PrimateII::MO_LO);
    SDValue MNHi = SDValue(DAG.getMachineNode(Primate::LUI, DL, Ty, AddrHi), 0);
    return SDValue(DAG.getMachineNode(Primate::ADDI, DL, Ty, MNHi, AddrLo), 0);
  }
  case CodeModel::Medium: {
    // Generate a sequence for accessing addresses within any 2GiB range within
    // the address space. This generates the pattern (PseudoLLA sym), which
    // expands to (addi (auipc %pcrel_hi(sym)) %pcrel_lo(auipc)).
    SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);
    return SDValue(DAG.getMachineNode(Primate::PseudoLLA, DL, Ty, Addr), 0);
  }
  }
}

SDValue PrimateTargetLowering::lowerGlobalAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  const GlobalValue *GV = N->getGlobal();
  bool IsLocal = getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);
  SDValue Addr = getAddr(N, DAG, IsLocal);

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, XLenVT));
  return Addr;
}

SDValue PrimateTargetLowering::lowerBlockAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  BlockAddressSDNode *N = cast<BlockAddressSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue PrimateTargetLowering::lowerConstantPool(SDValue Op,
                                               SelectionDAG &DAG) const {
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue PrimateTargetLowering::lowerJumpTable(SDValue Op,
                                            SelectionDAG &DAG) const {
  JumpTableSDNode *N = cast<JumpTableSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue PrimateTargetLowering::getStaticTLSAddr(GlobalAddressSDNode *N,
                                              SelectionDAG &DAG,
                                              bool UseGOT) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  const GlobalValue *GV = N->getGlobal();
  MVT XLenVT = Subtarget.getXLenVT();

  if (UseGOT) {
    // Use PC-relative addressing to access the GOT for this TLS symbol, then
    // load the address from the GOT and add the thread pointer. This generates
    // the pattern (PseudoLA_TLS_IE sym), which expands to
    // (ld (auipc %tls_ie_pcrel_hi(sym)) %pcrel_lo(auipc)).
    SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, Ty, 0, 0);
    SDValue Load =
        SDValue(DAG.getMachineNode(Primate::PseudoLA_TLS_IE, DL, Ty, Addr), 0);

    // Add the thread pointer.
    SDValue TPReg = DAG.getRegister(Primate::X4, XLenVT);
    return DAG.getNode(ISD::ADD, DL, Ty, Load, TPReg);
  }

  // Generate a sequence for accessing the address relative to the thread
  // pointer, with the appropriate adjustment for the thread pointer offset.
  // This generates the pattern
  // (add (add_tprel (lui %tprel_hi(sym)) tp %tprel_add(sym)) %tprel_lo(sym))
  SDValue AddrHi =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, PrimateII::MO_TPREL_HI);
  SDValue AddrAdd =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, PrimateII::MO_TPREL_ADD);
  SDValue AddrLo =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, PrimateII::MO_TPREL_LO);

  SDValue MNHi = SDValue(DAG.getMachineNode(Primate::LUI, DL, Ty, AddrHi), 0);
  SDValue TPReg = DAG.getRegister(Primate::X4, XLenVT);
  SDValue MNAdd = SDValue(
      DAG.getMachineNode(Primate::PseudoAddTPRel, DL, Ty, MNHi, TPReg, AddrAdd),
      0);
  return SDValue(DAG.getMachineNode(Primate::ADDI, DL, Ty, MNAdd, AddrLo), 0);
}

SDValue PrimateTargetLowering::getDynamicTLSAddr(GlobalAddressSDNode *N,
                                               SelectionDAG &DAG) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  IntegerType *CallTy = Type::getIntNTy(*DAG.getContext(), Ty.getSizeInBits());
  const GlobalValue *GV = N->getGlobal();

  // Use a PC-relative addressing mode to access the global dynamic GOT address.
  // This generates the pattern (PseudoLA_TLS_GD sym), which expands to
  // (addi (auipc %tls_gd_pcrel_hi(sym)) %pcrel_lo(auipc)).
  SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, Ty, 0, 0);
  SDValue Load =
      SDValue(DAG.getMachineNode(Primate::PseudoLA_TLS_GD, DL, Ty, Addr), 0);

  // Prepare argument list to generate call.
  ArgListTy Args;
  ArgListEntry Entry;
  Entry.Node = Load;
  Entry.Ty = CallTy;
  Args.push_back(Entry);

  // Setup call to __tls_get_addr.
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(DL)
      .setChain(DAG.getEntryNode())
      .setLibCallee(CallingConv::C, CallTy,
                    DAG.getExternalSymbol("__tls_get_addr", Ty),
                    std::move(Args));

  return LowerCallTo(CLI).first;
}

SDValue PrimateTargetLowering::lowerGlobalTLSAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  TLSModel::Model Model = getTargetMachine().getTLSModel(N->getGlobal());

  if (DAG.getMachineFunction().getFunction().getCallingConv() ==
      CallingConv::GHC)
    report_fatal_error("In GHC calling convention TLS is not supported");

  SDValue Addr;
  switch (Model) {
  case TLSModel::LocalExec:
    Addr = getStaticTLSAddr(N, DAG, /*UseGOT=*/false);
    break;
  case TLSModel::InitialExec:
    Addr = getStaticTLSAddr(N, DAG, /*UseGOT=*/true);
    break;
  case TLSModel::LocalDynamic:
  case TLSModel::GeneralDynamic:
    Addr = getDynamicTLSAddr(N, DAG);
    break;
  }

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, XLenVT));
  return Addr;
}

SDValue PrimateTargetLowering::lowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  SDValue CondV = Op.getOperand(0);
  SDValue TrueV = Op.getOperand(1);
  SDValue FalseV = Op.getOperand(2);
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  // Lower vector SELECTs to VSELECTs by splatting the condition.
  if (VT.isVector()) {
    MVT SplatCondVT = VT.changeVectorElementType(MVT::i1);
    SDValue CondSplat = VT.isScalableVector()
                            ? DAG.getSplatVector(SplatCondVT, DL, CondV)
                            : DAG.getSplatBuildVector(SplatCondVT, DL, CondV);
    return DAG.getNode(ISD::VSELECT, DL, VT, CondSplat, TrueV, FalseV);
  }

  // If the result type is XLenVT and CondV is the output of a SETCC node
  // which also operated on XLenVT inputs, then merge the SETCC node into the
  // lowered PrimateISD::SELECT_CC to take advantage of the integer
  // compare+branch instructions. i.e.:
  // (select (setcc lhs, rhs, cc), truev, falsev)
  // -> (primateisd::select_cc lhs, rhs, cc, truev, falsev)
  if (VT == XLenVT && CondV.getOpcode() == ISD::SETCC &&
      CondV.getOperand(0).getSimpleValueType() == XLenVT) {
    SDValue LHS = CondV.getOperand(0);
    SDValue RHS = CondV.getOperand(1);
    const auto *CC = cast<CondCodeSDNode>(CondV.getOperand(2));
    ISD::CondCode CCVal = CC->get();

    // Special case for a select of 2 constants that have a diffence of 1.
    // Normally this is done by DAGCombine, but if the select is introduced by
    // type legalization or op legalization, we miss it. Restricting to SETLT
    // case for now because that is what signed saturating add/sub need.
    // FIXME: We don't need the condition to be SETLT or even a SETCC,
    // but we would probably want to swap the true/false values if the condition
    // is SETGE/SETLE to avoid an XORI.
    if (isa<ConstantSDNode>(TrueV) && isa<ConstantSDNode>(FalseV) &&
        CCVal == ISD::SETLT) {
      const APInt &TrueVal = cast<ConstantSDNode>(TrueV)->getAPIntValue();
      const APInt &FalseVal = cast<ConstantSDNode>(FalseV)->getAPIntValue();
      if (TrueVal - 1 == FalseVal)
        return DAG.getNode(ISD::ADD, DL, Op.getValueType(), CondV, FalseV);
      if (TrueVal + 1 == FalseVal)
        return DAG.getNode(ISD::SUB, DL, Op.getValueType(), FalseV, CondV);
    }

    translateSetCCForBranch(DL, LHS, RHS, CCVal, DAG);

    SDValue TargetCC = DAG.getTargetConstant(CCVal, DL, XLenVT);
    SDValue Ops[] = {LHS, RHS, TargetCC, TrueV, FalseV};
    return DAG.getNode(PrimateISD::SELECT_CC, DL, Op.getValueType(), Ops);
  }

  // Otherwise:
  // (select condv, truev, falsev)
  // -> (primateisd::select_cc condv, zero, setne, truev, falsev)
  SDValue Zero = DAG.getConstant(0, DL, XLenVT);
  SDValue SetNE = DAG.getTargetConstant(ISD::SETNE, DL, XLenVT);

  SDValue Ops[] = {CondV, Zero, SetNE, TrueV, FalseV};

  return DAG.getNode(PrimateISD::SELECT_CC, DL, Op.getValueType(), Ops);
}

SDValue PrimateTargetLowering::lowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  SDValue CondV = Op.getOperand(1);
  SDLoc DL(Op);
  MVT XLenVT = Subtarget.getXLenVT();

  if (CondV.getOpcode() == ISD::SETCC &&
      CondV.getOperand(0).getValueType() == XLenVT) {
    SDValue LHS = CondV.getOperand(0);
    SDValue RHS = CondV.getOperand(1);
    ISD::CondCode CCVal = cast<CondCodeSDNode>(CondV.getOperand(2))->get();

    translateSetCCForBranch(DL, LHS, RHS, CCVal, DAG);

    SDValue TargetCC = DAG.getCondCode(CCVal);
    return DAG.getNode(PrimateISD::BR_CC, DL, Op.getValueType(), Op.getOperand(0),
                       LHS, RHS, TargetCC, Op.getOperand(2));
  }

  return DAG.getNode(PrimateISD::BR_CC, DL, Op.getValueType(), Op.getOperand(0),
                     CondV, DAG.getConstant(0, DL, XLenVT),
                     DAG.getCondCode(ISD::SETNE), Op.getOperand(2));
}

SDValue PrimateTargetLowering::lowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  PrimateMachineFunctionInfo *FuncInfo = MF.getInfo<PrimateMachineFunctionInfo>();

  SDLoc DL(Op);
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy(MF.getDataLayout()));

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, FI, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue PrimateTargetLowering::lowerFRAMEADDR(SDValue Op,
                                            SelectionDAG &DAG) const {
  const PrimateRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setFrameAddressIsTaken(true);
  Register FrameReg = RI.getFrameRegister(MF);
  int XLenInBytes = Subtarget.getXLen() / 8;

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), DL, FrameReg, VT);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  while (Depth--) {
    int Offset = -(XLenInBytes * 2);
    SDValue Ptr = DAG.getNode(ISD::ADD, DL, VT, FrameAddr,
                              DAG.getIntPtrConstant(Offset, DL));
    FrameAddr =
        DAG.getLoad(VT, DL, DAG.getEntryNode(), Ptr, MachinePointerInfo());
  }
  return FrameAddr;
}

SDValue PrimateTargetLowering::lowerRETURNADDR(SDValue Op,
                                             SelectionDAG &DAG) const {
  const PrimateRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setReturnAddressIsTaken(true);
  MVT XLenVT = Subtarget.getXLenVT();
  int XLenInBytes = Subtarget.getXLen() / 8;

  if (verifyReturnAddressArgumentIsConstant(Op, DAG))
    return SDValue();

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (Depth) {
    int Off = -XLenInBytes;
    SDValue FrameAddr = lowerFRAMEADDR(Op, DAG);
    SDValue Offset = DAG.getConstant(Off, DL, VT);
    return DAG.getLoad(VT, DL, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, DL, VT, FrameAddr, Offset),
                       MachinePointerInfo());
  }

  // Return the value of the return address register, marking it an implicit
  // live-in.
  Register Reg = MF.addLiveIn(RI.getRARegister(), getRegClassFor(XLenVT));
  return DAG.getCopyFromReg(DAG.getEntryNode(), DL, Reg, XLenVT);
}

SDValue PrimateTargetLowering::lowerShiftLeftParts(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // if Shamt-XLEN < 0: // Shamt < XLEN
  //   Lo = Lo << Shamt
  //   Hi = (Hi << Shamt) | ((Lo >>u 1) >>u (XLEN-1 - Shamt))
  // else:
  //   Lo = 0
  //   Hi = Lo << (Shamt-XLEN)

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusXLen = DAG.getConstant(-(int)Subtarget.getXLen(), DL, VT);
  SDValue XLenMinus1 = DAG.getConstant(Subtarget.getXLen() - 1, DL, VT);
  SDValue ShamtMinusXLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusXLen);
  SDValue XLenMinus1Shamt = DAG.getNode(ISD::SUB, DL, VT, XLenMinus1, Shamt);

  SDValue LoTrue = DAG.getNode(ISD::SHL, DL, VT, Lo, Shamt);
  SDValue ShiftRight1Lo = DAG.getNode(ISD::SRL, DL, VT, Lo, One);
  SDValue ShiftRightLo =
      DAG.getNode(ISD::SRL, DL, VT, ShiftRight1Lo, XLenMinus1Shamt);
  SDValue ShiftLeftHi = DAG.getNode(ISD::SHL, DL, VT, Hi, Shamt);
  SDValue HiTrue = DAG.getNode(ISD::OR, DL, VT, ShiftLeftHi, ShiftRightLo);
  SDValue HiFalse = DAG.getNode(ISD::SHL, DL, VT, Lo, ShamtMinusXLen);

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusXLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, Zero);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

SDValue PrimateTargetLowering::lowerShiftRightParts(SDValue Op, SelectionDAG &DAG,
                                                  bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // SRA expansion:
  //   if Shamt-XLEN < 0: // Shamt < XLEN
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (XLEN-1 - Shamt))
  //     Hi = Hi >>s Shamt
  //   else:
  //     Lo = Hi >>s (Shamt-XLEN);
  //     Hi = Hi >>s (XLEN-1)
  //
  // SRL expansion:
  //   if Shamt-XLEN < 0: // Shamt < XLEN
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (XLEN-1 - Shamt))
  //     Hi = Hi >>u Shamt
  //   else:
  //     Lo = Hi >>u (Shamt-XLEN);
  //     Hi = 0;

  unsigned ShiftRightOp = IsSRA ? ISD::SRA : ISD::SRL;

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusXLen = DAG.getConstant(-(int)Subtarget.getXLen(), DL, VT);
  SDValue XLenMinus1 = DAG.getConstant(Subtarget.getXLen() - 1, DL, VT);
  SDValue ShamtMinusXLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusXLen);
  SDValue XLenMinus1Shamt = DAG.getNode(ISD::SUB, DL, VT, XLenMinus1, Shamt);

  SDValue ShiftRightLo = DAG.getNode(ISD::SRL, DL, VT, Lo, Shamt);
  SDValue ShiftLeftHi1 = DAG.getNode(ISD::SHL, DL, VT, Hi, One);
  SDValue ShiftLeftHi =
      DAG.getNode(ISD::SHL, DL, VT, ShiftLeftHi1, XLenMinus1Shamt);
  SDValue LoTrue = DAG.getNode(ISD::OR, DL, VT, ShiftRightLo, ShiftLeftHi);
  SDValue HiTrue = DAG.getNode(ShiftRightOp, DL, VT, Hi, Shamt);
  SDValue LoFalse = DAG.getNode(ShiftRightOp, DL, VT, Hi, ShamtMinusXLen);
  SDValue HiFalse =
      IsSRA ? DAG.getNode(ISD::SRA, DL, VT, Hi, XLenMinus1) : Zero;

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusXLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, LoFalse);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

// Lower splats of i1 types to SETCC. For each mask vector type, we have a
// legal equivalently-sized i8 type, so we can use that as a go-between.
SDValue PrimateTargetLowering::lowerVectorMaskSplat(SDValue Op,
                                                  SelectionDAG &DAG) const {
  llvm_unreachable("Primate LowerVectorMaskSplit");
  // SDLoc DL(Op);
  // MVT VT = Op.getSimpleValueType();
  // SDValue SplatVal = Op.getOperand(0);
  // // All-zeros or all-ones splats are handled specially.
  // if (ISD::isConstantSplatVectorAllOnes(Op.getNode())) {
  //   SDValue VL = getDefaultScalableVLOps(VT, DL, DAG, Subtarget).second;
  //   return DAG.getNode(PrimateISD::VMSET_VL, DL, VT, VL);
  // }
  // if (ISD::isConstantSplatVectorAllZeros(Op.getNode())) {
  //   SDValue VL = getDefaultScalableVLOps(VT, DL, DAG, Subtarget).second;
  //   return DAG.getNode(PrimateISD::VMCLR_VL, DL, VT, VL);
  // }
  // MVT XLenVT = Subtarget.getXLenVT();
  // assert(SplatVal.getValueType() == XLenVT &&
  //        "Unexpected type for i1 splat value");
  // MVT InterVT = VT.changeVectorElementType(MVT::i8);
  // SplatVal = DAG.getNode(ISD::AND, DL, XLenVT, SplatVal,
  //                        DAG.getConstant(1, DL, XLenVT));
  // SDValue LHS = DAG.getSplatVector(InterVT, DL, SplatVal);
  // SDValue Zero = DAG.getConstant(0, DL, InterVT);
  // return DAG.getSetCC(DL, VT, LHS, Zero, ISD::SETNE);
}

// Custom-lower a SPLAT_VECTOR_PARTS where XLEN<SEW, as the SEW element type is
// illegal (currently only vXi64 PR32).
// FIXME: We could also catch non-constant sign-extended i32 values and lower
// them to SPLAT_VECTOR_I64
SDValue PrimateTargetLowering::lowerSPLAT_VECTOR_PARTS(SDValue Op,
                                                     SelectionDAG &DAG) const {
  llvm_unreachable("Primate lowerSPLAT_VECTOR_PARTS");
  // SDLoc DL(Op);
  // MVT VecVT = Op.getSimpleValueType();
  // assert(!Subtarget.is64Bit() && VecVT.getVectorElementType() == MVT::i64 &&
  //        "Unexpected SPLAT_VECTOR_PARTS lowering");

  // assert(Op.getNumOperands() == 2 && "Unexpected number of operands!");
  // SDValue Lo = Op.getOperand(0);
  // SDValue Hi = Op.getOperand(1);

  // if (VecVT.isFixedLengthVector()) {
  //   MVT ContainerVT = getContainerForFixedLengthVector(VecVT);
  //   SDLoc DL(Op);
  //   SDValue Mask, VL;
  //   std::tie(Mask, VL) =
  //       getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  //   SDValue Res = splatPartsI64WithVL(DL, ContainerVT, Lo, Hi, VL, DAG);
  //   return convertFromScalableVector(VecVT, Res, DAG, Subtarget);
  // }

  // if (isa<ConstantSDNode>(Lo) && isa<ConstantSDNode>(Hi)) {
  //   int32_t LoC = cast<ConstantSDNode>(Lo)->getSExtValue();
  //   int32_t HiC = cast<ConstantSDNode>(Hi)->getSExtValue();
  //   // If Hi constant is all the same sign bit as Lo, lower this as a custom
  //   // node in order to try and match PRV vector/scalar instructions.
  //   if ((LoC >> 31) == HiC)
  //     return DAG.getNode(PrimateISD::SPLAT_VECTOR_I64, DL, VecVT, Lo);
  // }

  // // Detect cases where Hi is (SRA Lo, 31) which means Hi is Lo sign extended.
  // if (Hi.getOpcode() == ISD::SRA && Hi.getOperand(0) == Lo &&
  //     isa<ConstantSDNode>(Hi.getOperand(1)) &&
  //     Hi.getConstantOperandVal(1) == 31)
  //   return DAG.getNode(PrimateISD::SPLAT_VECTOR_I64, DL, VecVT, Lo);

  // // Fall back to use a stack store and stride x0 vector load. Use X0 as VL.
  // return DAG.getNode(PrimateISD::SPLAT_VECTOR_SPLIT_I64_VL, DL, VecVT, Lo, Hi,
  //                    DAG.getRegister(Primate::X0, MVT::i64));
}

// Custom-lower extensions from mask vectors by using a vselect either with 1
// for zero/any-extension or -1 for sign-extension:
//   (vXiN = (s|z)ext vXi1:vmask) -> (vXiN = vselect vmask, (-1 or 1), 0)
// Note that any-extension is lowered identically to zero-extension.
SDValue PrimateTargetLowering::lowerVectorMaskExt(SDValue Op, SelectionDAG &DAG,
                                                int64_t ExtTrueVal) const {
  llvm_unreachable("Primate lowerVectorMaskExt");
  // SDLoc DL(Op);
  // MVT VecVT = Op.getSimpleValueType();
  // SDValue Src = Op.getOperand(0);
  // // Only custom-lower extensions from mask types
  // assert(Src.getValueType().isVector() &&
  //        Src.getValueType().getVectorElementType() == MVT::i1);

  // MVT XLenVT = Subtarget.getXLenVT();
  // SDValue SplatZero = DAG.getConstant(0, DL, XLenVT);
  // SDValue SplatTrueVal = DAG.getConstant(ExtTrueVal, DL, XLenVT);

  // if (VecVT.isScalableVector()) {
  //   // Be careful not to introduce illegal scalar types at this stage, and be
  //   // careful also about splatting constants as on PR32, vXi64 SPLAT_VECTOR is
  //   // illegal and must be expanded. Since we know that the constants are
  //   // sign-extended 32-bit values, we use SPLAT_VECTOR_I64 directly.
  //   bool IsPR32E64 =
  //       !Subtarget.is64Bit() && VecVT.getVectorElementType() == MVT::i64;

  //   if (!IsPR32E64) {
  //     SplatZero = DAG.getSplatVector(VecVT, DL, SplatZero);
  //     SplatTrueVal = DAG.getSplatVector(VecVT, DL, SplatTrueVal);
  //   } else {
  //     SplatZero = DAG.getNode(PrimateISD::SPLAT_VECTOR_I64, DL, VecVT, SplatZero);
  //     SplatTrueVal =
  //         DAG.getNode(PrimateISD::SPLAT_VECTOR_I64, DL, VecVT, SplatTrueVal);
  //   }

  //   return DAG.getNode(ISD::VSELECT, DL, VecVT, Src, SplatTrueVal, SplatZero);
  // }

  // MVT ContainerVT = getContainerForFixedLengthVector(VecVT);
  // MVT I1ContainerVT =
  //     MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());

  // SDValue CC = convertToScalableVector(I1ContainerVT, Src, DAG, Subtarget);

  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  // SplatZero = DAG.getNode(PrimateISD::VMV_V_X_VL, DL, ContainerVT, SplatZero, VL);
  // SplatTrueVal =
  //     DAG.getNode(PrimateISD::VMV_V_X_VL, DL, ContainerVT, SplatTrueVal, VL);
  // SDValue Select = DAG.getNode(PrimateISD::VSELECT_VL, DL, ContainerVT, CC,
  //                              SplatTrueVal, SplatZero, VL);

  // return convertFromScalableVector(VecVT, Select, DAG, Subtarget);
}

SDValue PrimateTargetLowering::lowerFixedLengthVectorExtendToPRV(
    SDValue Op, SelectionDAG &DAG, unsigned ExtendOpc) const {
  MVT ExtVT = Op.getSimpleValueType();
  // Only custom-lower extensions from fixed-length vector types.
  if (!ExtVT.isFixedLengthVector())
    return Op;
  MVT VT = Op.getOperand(0).getSimpleValueType();
  // Grab the canonical container type for the extended type. Infer the smaller
  // type from that to ensure the same number of vector elements, as we know
  // the LMUL will be sufficient to hold the smaller type.
  MVT ContainerExtVT = getContainerForFixedLengthVector(ExtVT);
  // Get the extended container type manually to ensure the same number of
  // vector elements between source and dest.
  MVT ContainerVT = MVT::getVectorVT(VT.getVectorElementType(),
                                     ContainerExtVT.getVectorElementCount());

  SDValue Op1 =
      convertToScalableVector(ContainerVT, Op.getOperand(0), DAG, Subtarget);

  SDLoc DL(Op);
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  SDValue Ext = DAG.getNode(ExtendOpc, DL, ContainerExtVT, Op1, Mask, VL);

  return convertFromScalableVector(ExtVT, Ext, DAG, Subtarget);
}

// Custom-lower truncations from vectors to mask vectors by using a mask and a
// setcc operation:
//   (vXi1 = trunc vXiN vec) -> (vXi1 = setcc (and vec, 1), 0, ne)
SDValue PrimateTargetLowering::lowerVectorMaskTrunc(SDValue Op,
                                                  SelectionDAG &DAG) const {
  llvm_unreachable("Primate lowerVectorMaskTrunc");
  // SDLoc DL(Op);
  // EVT MaskVT = Op.getValueType();
  // // Only expect to custom-lower truncations to mask types
  // assert(MaskVT.isVector() && MaskVT.getVectorElementType() == MVT::i1 &&
  //        "Unexpected type for vector mask lowering");
  // SDValue Src = Op.getOperand(0);
  // MVT VecVT = Src.getSimpleValueType();

  // // If this is a fixed vector, we need to convert it to a scalable vector.
  // MVT ContainerVT = VecVT;
  // if (VecVT.isFixedLengthVector()) {
  //   ContainerVT = getContainerForFixedLengthVector(VecVT);
  //   Src = convertToScalableVector(ContainerVT, Src, DAG, Subtarget);
  // }

  // SDValue SplatOne = DAG.getConstant(1, DL, Subtarget.getXLenVT());
  // SDValue SplatZero = DAG.getConstant(0, DL, Subtarget.getXLenVT());

  // SplatOne = DAG.getNode(PrimateISD::VMV_V_X_VL, DL, ContainerVT, SplatOne);
  // SplatZero = DAG.getNode(PrimateISD::VMV_V_X_VL, DL, ContainerVT, SplatZero);

  // if (VecVT.isScalableVector()) {
  //   SDValue Trunc = DAG.getNode(ISD::AND, DL, VecVT, Src, SplatOne);
  //   return DAG.getSetCC(DL, MaskVT, Trunc, SplatZero, ISD::SETNE);
  // }

  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  // MVT MaskContainerVT = ContainerVT.changeVectorElementType(MVT::i1);
  // SDValue Trunc =
  //     DAG.getNode(PrimateISD::AND_VL, DL, ContainerVT, Src, SplatOne, Mask, VL);
  // Trunc = DAG.getNode(PrimateISD::SETCC_VL, DL, MaskContainerVT, Trunc, SplatZero,
  //                     DAG.getCondCode(ISD::SETNE), Mask, VL);
  // return convertFromScalableVector(MaskVT, Trunc, DAG, Subtarget);
}

// Custom-legalize INSERT_VECTOR_ELT so that the value is inserted into the
// first position of a vector, and that vector is slid up to the insert index.
// By limiting the active vector length to index+1 and merging with the
// original vector (with an undisturbed tail policy for elements >= VL), we
// achieve the desired result of leaving all elements untouched except the one
// at VL-1, which is replaced with the desired value.
SDValue PrimateTargetLowering::lowerINSERT_VECTOR_ELT(SDValue Op,
                                                    SelectionDAG &DAG) const {
  llvm_unreachable("Primate Lower Insert Vector Elt");
  // SDLoc DL(Op);
  // MVT VecVT = Op.getSimpleValueType();
  // SDValue Vec = Op.getOperand(0);
  // SDValue Val = Op.getOperand(1);
  // SDValue Idx = Op.getOperand(2);

  // if (VecVT.getVectorElementType() == MVT::i1) {
  //   // FIXME: For now we just promote to an i8 vector and insert into that,
  //   // but this is probably not optimal.
  //   MVT WideVT = MVT::getVectorVT(MVT::i8, VecVT.getVectorElementCount());
  //   Vec = DAG.getNode(ISD::ZERO_EXTEND, DL, WideVT, Vec);
  //   Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, WideVT, Vec, Val, Idx);
  //   return DAG.getNode(ISD::TRUNCATE, DL, VecVT, Vec);
  // }

  // MVT ContainerVT = VecVT;
  // // If the operand is a fixed-length vector, convert to a scalable one.
  // if (VecVT.isFixedLengthVector()) {
  //   ContainerVT = getContainerForFixedLengthVector(VecVT);
  //   Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  // }

  // MVT XLenVT = Subtarget.getXLenVT();

  // SDValue Zero = DAG.getConstant(0, DL, XLenVT);
  // bool IsLegalInsert = Subtarget.is64Bit() || Val.getValueType() != MVT::i64;
  // // Even i64-element vectors on PR32 can be lowered without scalar
  // // legalization if the most-significant 32 bits of the value are not affected
  // // by the sign-extension of the lower 32 bits.
  // // TODO: We could also catch sign extensions of a 32-bit value.
  // if (!IsLegalInsert && isa<ConstantSDNode>(Val)) {
  //   const auto *CVal = cast<ConstantSDNode>(Val);
  //   if (isInt<32>(CVal->getSExtValue())) {
  //     IsLegalInsert = true;
  //     Val = DAG.getConstant(CVal->getSExtValue(), DL, MVT::i32);
  //   }
  // }

  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  // SDValue ValInVec;

  // if (IsLegalInsert) {
  //   unsigned Opc =
  //       VecVT.isFloatingPoint() ? PrimateISD::VFMV_S_F_VL : PrimateISD::VMV_S_X_VL;
  //   if (isNullConstant(Idx)) {
  //     Vec = DAG.getNode(Opc, DL, ContainerVT, Vec, Val, VL);
  //     if (!VecVT.isFixedLengthVector())
  //       return Vec;
  //     return convertFromScalableVector(VecVT, Vec, DAG, Subtarget);
  //   }
  //   ValInVec =
  //       DAG.getNode(Opc, DL, ContainerVT, DAG.getUNDEF(ContainerVT), Val, VL);
  // } else {
  //   // On PR32, i64-element vectors must be specially handled to place the
  //   // value at element 0, by using two vslide1up instructions in sequence on
  //   // the i32 split lo/hi value. Use an equivalently-sized i32 vector for
  //   // this.
  //   SDValue One = DAG.getConstant(1, DL, XLenVT);
  //   SDValue ValLo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Val, Zero);
  //   SDValue ValHi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Val, One);
  //   MVT I32ContainerVT =
  //       MVT::getVectorVT(MVT::i32, ContainerVT.getVectorElementCount() * 2);
  //   SDValue I32Mask =
  //       getDefaultScalableVLOps(I32ContainerVT, DL, DAG, Subtarget).first;
  //   // Limit the active VL to two.
  //   SDValue InsertI64VL = DAG.getConstant(2, DL, XLenVT);
  //   // Note: We can't pass a UNDEF to the first VSLIDE1UP_VL since an untied
  //   // undef doesn't obey the earlyclobber constraint. Just splat a zero value.
  //   ValInVec = DAG.getNode(PrimateISD::VMV_V_X_VL, DL, I32ContainerVT, Zero,
  //                          InsertI64VL);
  //   // First slide in the hi value, then the lo in underneath it.
  //   ValInVec = DAG.getNode(PrimateISD::VSLIDE1UP_VL, DL, I32ContainerVT, ValInVec,
  //                          ValHi, I32Mask, InsertI64VL);
  //   ValInVec = DAG.getNode(PrimateISD::VSLIDE1UP_VL, DL, I32ContainerVT, ValInVec,
  //                          ValLo, I32Mask, InsertI64VL);
  //   // Bitcast back to the right container type.
  //   ValInVec = DAG.getBitcast(ContainerVT, ValInVec);
  // }

  // // Now that the value is in a vector, slide it into position.
  // SDValue InsertVL =
  //     DAG.getNode(ISD::ADD, DL, XLenVT, Idx, DAG.getConstant(1, DL, XLenVT));
  // SDValue Slideup = DAG.getNode(PrimateISD::VSLIDEUP_VL, DL, ContainerVT, Vec,
  //                               ValInVec, Idx, Mask, InsertVL);
  // if (!VecVT.isFixedLengthVector())
  //   return Slideup;
  // return convertFromScalableVector(VecVT, Slideup, DAG, Subtarget);
}

// Custom-lower EXTRACT_VECTOR_ELT operations to slide the vector down, then
// extract the first element: (extractelt (slidedown vec, idx), 0). For integer
// types this is done using VMV_X_S to allow us to glean information about the
// sign bits of the result.
SDValue PrimateTargetLowering::lowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                     SelectionDAG &DAG) const {
  llvm_unreachable("Primate lowerEXTRACT_VECTOR_ELT");
  // SDLoc DL(Op);
  // SDValue Idx = Op.getOperand(1);
  // SDValue Vec = Op.getOperand(0);
  // EVT EltVT = Op.getValueType();
  // MVT VecVT = Vec.getSimpleValueType();
  // MVT XLenVT = Subtarget.getXLenVT();

  // if (VecVT.getVectorElementType() == MVT::i1) {
  //   // FIXME: For now we just promote to an i8 vector and extract from that,
  //   // but this is probably not optimal.
  //   MVT WideVT = MVT::getVectorVT(MVT::i8, VecVT.getVectorElementCount());
  //   Vec = DAG.getNode(ISD::ZERO_EXTEND, DL, WideVT, Vec);
  //   return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Vec, Idx);
  // }

  // // If this is a fixed vector, we need to convert it to a scalable vector.
  // MVT ContainerVT = VecVT;
  // if (VecVT.isFixedLengthVector()) {
  //   ContainerVT = getContainerForFixedLengthVector(VecVT);
  //   Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  // }

  // // If the index is 0, the vector is already in the right position.
  // if (!isNullConstant(Idx)) {
  //   // Use a VL of 1 to avoid processing more elements than we need.
  //   SDValue VL = DAG.getConstant(1, DL, XLenVT);
  //   MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
  //   SDValue Mask = DAG.getNode(PrimateISD::VMSET_VL, DL, MaskVT, VL);
  //   Vec = DAG.getNode(PrimateISD::VSLIDEDOWN_VL, DL, ContainerVT,
  //                     DAG.getUNDEF(ContainerVT), Vec, Idx, Mask, VL);
  // }

  // if (!EltVT.isInteger()) {
  //   // Floating-point extracts are handled in TableGen.
  //   return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Vec,
  //                      DAG.getConstant(0, DL, XLenVT));
  // }

  // SDValue Elt0 = DAG.getNode(PrimateISD::VMV_X_S, DL, XLenVT, Vec);
  // return DAG.getNode(ISD::TRUNCATE, DL, EltVT, Elt0);
}

// Some PRV intrinsics may claim that they want an integer operand to be
// promoted or expanded.
static SDValue lowerVectorIntrinsicSplats(SDValue Op, SelectionDAG &DAG,
                                          const PrimateSubtarget &Subtarget) {
  llvm_unreachable("Primate lowering vectorIntrinsicSplats");
  // assert((Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
  //         Op.getOpcode() == ISD::INTRINSIC_W_CHAIN) &&
  //        "Unexpected opcode");
  // dbgs() << "Something thinks its a vector intrinsic splat\n";

  // if (!Subtarget.hasStdExtV())
  //   return SDValue();

  // bool HasChain = Op.getOpcode() == ISD::INTRINSIC_W_CHAIN;
  // unsigned IntNo = Op.getConstantOperandVal(HasChain ? 1 : 0);
  // SDLoc DL(Op);

  // const PrimateVIntrinsicsTable::PrimateVIntrinsicInfo *II =
  //     PrimateVIntrinsicsTable::getPrimateVIntrinsicInfo(IntNo);
  // if (!II || !II->SplatOperand)
  //   return SDValue();

  // unsigned SplatOp = II->SplatOperand + HasChain;
  // assert(SplatOp < Op.getNumOperands());

  // SmallVector<SDValue, 8> Operands(Op->op_begin(), Op->op_end());
  // SDValue &ScalarOp = Operands[SplatOp];
  // MVT OpVT = ScalarOp.getSimpleValueType();
  // MVT XLenVT = Subtarget.getXLenVT();

  // // If this isn't a scalar, or its type is XLenVT we're done.
  // if (!OpVT.isScalarInteger() || OpVT == XLenVT)
  //   return SDValue();

  // // Simplest case is that the operand needs to be promoted to XLenVT.
  // if (OpVT.bitsLT(XLenVT)) {
  //   // If the operand is a constant, sign extend to increase our chances
  //   // of being able to use a .vi instruction. ANY_EXTEND would become a
  //   // a zero extend and the simm5 check in isel would fail.
  //   // FIXME: Should we ignore the upper bits in isel instead?
  //   unsigned ExtOpc =
  //       isa<ConstantSDNode>(ScalarOp) ? ISD::SIGN_EXTEND : ISD::ANY_EXTEND;
  //   ScalarOp = DAG.getNode(ExtOpc, DL, XLenVT, ScalarOp);
  //   return DAG.getNode(Op->getOpcode(), DL, Op->getVTList(), Operands);
  // }

  // // Use the previous operand to get the vXi64 VT. The result might be a mask
  // // VT for compares. Using the previous operand assumes that the previous
  // // operand will never have a smaller element size than a scalar operand and
  // // that a widening operation never uses SEW=64.
  // // NOTE: If this fails the below assert, we can probably just find the
  // // element count from any operand or result and use it to construct the VT.
  // assert(II->SplatOperand > 1 && "Unexpected splat operand!");
  // MVT VT = Op.getOperand(SplatOp - 1).getSimpleValueType();

  // // The more complex case is when the scalar is larger than XLenVT.
  // assert(XLenVT == MVT::i32 && OpVT == MVT::i64 &&
  //        VT.getVectorElementType() == MVT::i64 && "Unexpected VTs!");

  // // If this is a sign-extended 32-bit constant, we can truncate it and rely
  // // on the instruction to sign-extend since SEW>XLEN.
  // if (auto *CVal = dyn_cast<ConstantSDNode>(ScalarOp)) {
  //   if (isInt<32>(CVal->getSExtValue())) {
  //     ScalarOp = DAG.getConstant(CVal->getSExtValue(), DL, MVT::i32);
  //     return DAG.getNode(Op->getOpcode(), DL, Op->getVTList(), Operands);
  //   }
  // }

  // // We need to convert the scalar to a splat vector.
  // // FIXME: Can we implicitly truncate the scalar if it is known to
  // // be sign extended?
  // // VL should be the last operand.
  // SDValue VL = Op.getOperand(Op.getNumOperands() - 1);
  // assert(VL.getValueType() == XLenVT);
  // ScalarOp = splatSplitI64WithVL(DL, VT, ScalarOp, VL, DAG);
  // return DAG.getNode(Op->getOpcode(), DL, Op->getVTList(), Operands);
}

SDValue PrimateTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                     SelectionDAG &DAG) const {
  LLVM_DEBUG(dbgs() << "lower intrinsic wo chain while trying to construct dag\n");
  LLVM_DEBUG(Op.dump());
  LLVM_DEBUG(Op.getNode()->dump());

  unsigned IntNo = Op.getConstantOperandVal(0);
  SDLoc DL(Op);
  MVT XLenVT = Subtarget.getXLenVT();

  switch (IntNo) {
  default:
    break; // Don't custom lower most intrinsics.
  case Intrinsic::thread_pointer: {
    EVT PtrVT = getPointerTy(DAG.getDataLayout());
    return DAG.getRegister(Primate::X4, PtrVT);
  }
  case Intrinsic::primate_orc_b:
    // Lower to the GORCI encoding for orc.b.
    return DAG.getNode(PrimateISD::GORC, DL, XLenVT, Op.getOperand(1),
                       DAG.getConstant(7, DL, XLenVT));
  case Intrinsic::primate_grev:
  case Intrinsic::primate_gorc: {
    unsigned Opc =
        IntNo == Intrinsic::primate_grev ? PrimateISD::GREV : PrimateISD::GORC;
    return DAG.getNode(Opc, DL, XLenVT, Op.getOperand(1), Op.getOperand(2));
  }
  case Intrinsic::primate_shfl:
  case Intrinsic::primate_unshfl: {
    unsigned Opc =
        IntNo == Intrinsic::primate_shfl ? PrimateISD::SHFL : PrimateISD::UNSHFL;
    return DAG.getNode(Opc, DL, XLenVT, Op.getOperand(1), Op.getOperand(2));
  }
  case Intrinsic::primate_bcompress:
  case Intrinsic::primate_bdecompress: {
    unsigned Opc = IntNo == Intrinsic::primate_bcompress ? PrimateISD::BCOMPRESS
                                                       : PrimateISD::BDECOMPRESS;
    return DAG.getNode(Opc, DL, XLenVT, Op.getOperand(1), Op.getOperand(2));
  }
  case Intrinsic::primate_vmv_x_s:
    llvm_unreachable("primate custom lower vmx");
    // assert(Op.getValueType() == XLenVT && "Unexpected VT!");
    // return DAG.getNode(PrimateISD::VMV_X_S, DL, Op.getValueType(),
    //                    Op.getOperand(1));
  case Intrinsic::primate_vmv_v_x:
    llvm_unreachable("primate custom lower vmv");
    // return lowerScalarSplat(Op.getOperand(1), Op.getOperand(2),
    //                         Op.getSimpleValueType(), DL, DAG, Subtarget);
  case Intrinsic::primate_vfmv_v_f:
    llvm_unreachable("primate custom lower vfmv");
    // return DAG.getNode(PrimateISD::VFMV_V_F_VL, DL, Op.getValueType(),
    //                    Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::primate_vmv_s_x: {
    llvm_unreachable("primate custom lower vmv_s_x");
    // SDValue Scalar = Op.getOperand(2);

    // if (Scalar.getValueType().bitsLE(XLenVT)) {
    //   Scalar = DAG.getNode(ISD::ANY_EXTEND, DL, XLenVT, Scalar);
    //   return DAG.getNode(PrimateISD::VMV_S_X_VL, DL, Op.getValueType(),
    //                      Op.getOperand(1), Scalar, Op.getOperand(3));
    // }

    // assert(Scalar.getValueType() == MVT::i64 && "Unexpected scalar VT!");

    // // This is an i64 value that lives in two scalar registers. We have to
    // // insert this in a convoluted way. First we build vXi64 splat containing
    // // the/ two values that we assemble using some bit math. Next we'll use
    // // vid.v and vmseq to build a mask with bit 0 set. Then we'll use that mask
    // // to merge element 0 from our splat into the source vector.
    // // FIXME: This is probably not the best way to do this, but it is
    // // consistent with INSERT_VECTOR_ELT lowering so it is a good starting
    // // point.
    // //   sw lo, (a0)
    // //   sw hi, 4(a0)
    // //   vlse vX, (a0)
    // //
    // //   vid.v      vVid
    // //   vmseq.vx   mMask, vVid, 0
    // //   vmerge.vvm vDest, vSrc, vVal, mMask
    // MVT VT = Op.getSimpleValueType();
    // SDValue Vec = Op.getOperand(1);
    // SDValue VL = Op.getOperand(3);

    // SDValue SplattedVal = splatSplitI64WithVL(DL, VT, Scalar, VL, DAG);
    // SDValue SplattedIdx = DAG.getNode(PrimateISD::VMV_V_X_VL, DL, VT,
    //                                   DAG.getConstant(0, DL, MVT::i32), VL);

    // MVT MaskVT = MVT::getVectorVT(MVT::i1, VT.getVectorElementCount());
    // SDValue Mask = DAG.getNode(PrimateISD::VMSET_VL, DL, MaskVT, VL);
    // SDValue VID = DAG.getNode(PrimateISD::VID_VL, DL, VT, Mask, VL);
    // SDValue SelectCond =
    //     DAG.getNode(PrimateISD::SETCC_VL, DL, MaskVT, VID, SplattedIdx,
    //                 DAG.getCondCode(ISD::SETEQ), Mask, VL);
    // return DAG.getNode(PrimateISD::VSELECT_VL, DL, VT, SelectCond, SplattedVal,
    //                    Vec, VL);
  }
  case Intrinsic::primate_vslide1up:
  case Intrinsic::primate_vslide1down:
  case Intrinsic::primate_vslide1up_mask:
  case Intrinsic::primate_vslide1down_mask: {
    llvm_unreachable("primate custom lower vslide");
    // // We need to special case these when the scalar is larger than XLen.
    // unsigned NumOps = Op.getNumOperands();
    // bool IsMasked = NumOps == 6;
    // unsigned OpOffset = IsMasked ? 1 : 0;
    // SDValue Scalar = Op.getOperand(2 + OpOffset);
    // if (Scalar.getValueType().bitsLE(XLenVT))
    //   break;

    // // Splatting a sign extended constant is fine.
    // if (auto *CVal = dyn_cast<ConstantSDNode>(Scalar))
    //   if (isInt<32>(CVal->getSExtValue()))
    //     break;

    // MVT VT = Op.getSimpleValueType();
    // assert(VT.getVectorElementType() == MVT::i64 &&
    //        Scalar.getValueType() == MVT::i64 && "Unexpected VTs");

    // // Convert the vector source to the equivalent nxvXi32 vector.
    // MVT I32VT = MVT::getVectorVT(MVT::i32, VT.getVectorElementCount() * 2);
    // SDValue Vec = DAG.getBitcast(I32VT, Op.getOperand(1 + OpOffset));

    // SDValue ScalarLo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Scalar,
    //                                DAG.getConstant(0, DL, XLenVT));
    // SDValue ScalarHi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Scalar,
    //                                DAG.getConstant(1, DL, XLenVT));

    // // Double the VL since we halved SEW.
    // SDValue VL = Op.getOperand(NumOps - 1);
    // SDValue I32VL =
    //     DAG.getNode(ISD::SHL, DL, XLenVT, VL, DAG.getConstant(1, DL, XLenVT));

    // MVT I32MaskVT = MVT::getVectorVT(MVT::i1, I32VT.getVectorElementCount());
    // SDValue I32Mask = DAG.getNode(PrimateISD::VMSET_VL, DL, I32MaskVT, VL);

    // // Shift the two scalar parts in using SEW=32 slide1up/slide1down
    // // instructions.
    // if (IntNo == Intrinsic::primate_vslide1up ||
    //     IntNo == Intrinsic::primate_vslide1up_mask) {
    //   Vec = DAG.getNode(PrimateISD::VSLIDE1UP_VL, DL, I32VT, Vec, ScalarHi,
    //                     I32Mask, I32VL);
    //   Vec = DAG.getNode(PrimateISD::VSLIDE1UP_VL, DL, I32VT, Vec, ScalarLo,
    //                     I32Mask, I32VL);
    // } else {
    //   Vec = DAG.getNode(PrimateISD::VSLIDE1DOWN_VL, DL, I32VT, Vec, ScalarLo,
    //                     I32Mask, I32VL);
    //   Vec = DAG.getNode(PrimateISD::VSLIDE1DOWN_VL, DL, I32VT, Vec, ScalarHi,
    //                     I32Mask, I32VL);
    // }

    // // Convert back to nxvXi64.
    // Vec = DAG.getBitcast(VT, Vec);

    // if (!IsMasked)
    //   return Vec;

    // // Apply mask after the operation.
    // SDValue Mask = Op.getOperand(NumOps - 2);
    // SDValue MaskedOff = Op.getOperand(1);
    // return DAG.getNode(PrimateISD::VSELECT_VL, DL, VT, Mask, Vec, MaskedOff, VL);
  }
  }

  return lowerVectorIntrinsicSplats(Op, DAG, Subtarget);
}

SDValue PrimateTargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op,
                                                    SelectionDAG &DAG) const {
  dbgs() << "lowerINTRINSIC_W_CHAIN()\n";
  SDLoc DL(Op);
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  switch (IntNo) {
  default:
    return lowerVectorIntrinsicSplats(Op, DAG, Subtarget);
  }
}

static MVT getLMUL1VT(MVT VT) {
  assert(VT.getVectorElementType().getSizeInBits() <= 64 &&
         "Unexpected vector MVT");
  return MVT::getScalableVectorVT(
      VT.getVectorElementType(),
      Primate::PRVBitsPerBlock / VT.getVectorElementType().getSizeInBits());
}

static unsigned getPRVReductionOp(unsigned ISDOpcode) {
  switch (ISDOpcode) {
  default:
    llvm_unreachable("Unhandled reduction");
  // case ISD::VECREDUCE_ADD:
  //   return PrimateISD::VECREDUCE_ADD_VL;
  // case ISD::VECREDUCE_UMAX:
  //   return PrimateISD::VECREDUCE_UMAX_VL;
  // case ISD::VECREDUCE_SMAX:
  //   return PrimateISD::VECREDUCE_SMAX_VL;
  // case ISD::VECREDUCE_UMIN:
  //   return PrimateISD::VECREDUCE_UMIN_VL;
  // case ISD::VECREDUCE_SMIN:
  //   return PrimateISD::VECREDUCE_SMIN_VL;
  // case ISD::VECREDUCE_AND:
  //   return PrimateISD::VECREDUCE_AND_VL;
  // case ISD::VECREDUCE_OR:
  //   return PrimateISD::VECREDUCE_OR_VL;
  // case ISD::VECREDUCE_XOR:
  //   return PrimateISD::VECREDUCE_XOR_VL;
  }
}

SDValue PrimateTargetLowering::lowerVectorMaskVECREDUCE(SDValue Op,
                                                      SelectionDAG &DAG) const {
  llvm_unreachable("primate lowerVectorMaskVECREDUCE");
  // SDLoc DL(Op);
  // SDValue Vec = Op.getOperand(0);
  // MVT VecVT = Vec.getSimpleValueType();
  // assert((Op.getOpcode() == ISD::VECREDUCE_AND ||
  //         Op.getOpcode() == ISD::VECREDUCE_OR ||
  //         Op.getOpcode() == ISD::VECREDUCE_XOR) &&
  //        "Unexpected reduction lowering");

  // MVT XLenVT = Subtarget.getXLenVT();
  // assert(Op.getValueType() == XLenVT &&
  //        "Expected reduction output to be legalized to XLenVT");

  // MVT ContainerVT = VecVT;
  // if (VecVT.isFixedLengthVector()) {
  //   ContainerVT = getContainerForFixedLengthVector(VecVT);
  //   Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  // }

  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);
  // SDValue Zero = DAG.getConstant(0, DL, XLenVT);

  // switch (Op.getOpcode()) {
  // default:
  //   llvm_unreachable("Unhandled reduction");
  // case ISD::VECREDUCE_AND:
  //   // vpopc ~x == 0
  //   Vec = DAG.getNode(PrimateISD::VMXOR_VL, DL, ContainerVT, Vec, Mask, VL);
  //   Vec = DAG.getNode(PrimateISD::VPOPC_VL, DL, XLenVT, Vec, Mask, VL);
  //   return DAG.getSetCC(DL, XLenVT, Vec, Zero, ISD::SETEQ);
  // case ISD::VECREDUCE_OR:
  //   // vpopc x != 0
  //   Vec = DAG.getNode(PrimateISD::VPOPC_VL, DL, XLenVT, Vec, Mask, VL);
  //   return DAG.getSetCC(DL, XLenVT, Vec, Zero, ISD::SETNE);
  // case ISD::VECREDUCE_XOR: {
  //   // ((vpopc x) & 1) != 0
  //   SDValue One = DAG.getConstant(1, DL, XLenVT);
  //   Vec = DAG.getNode(PrimateISD::VPOPC_VL, DL, XLenVT, Vec, Mask, VL);
  //   Vec = DAG.getNode(ISD::AND, DL, XLenVT, Vec, One);
  //   return DAG.getSetCC(DL, XLenVT, Vec, Zero, ISD::SETNE);
  // }
  // }
}

SDValue PrimateTargetLowering::lowerVECREDUCE(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Vec = Op.getOperand(0);
  EVT VecEVT = Vec.getValueType();

  unsigned BaseOpc = ISD::getVecReduceBaseOpcode(Op.getOpcode());

  // Due to ordering in legalize types we may have a vector type that needs to
  // be split. Do that manually so we can get down to a legal type.
  while (getTypeAction(*DAG.getContext(), VecEVT) ==
         TargetLowering::TypeSplitVector) {
    SDValue Lo, Hi;
    std::tie(Lo, Hi) = DAG.SplitVector(Vec, DL);
    VecEVT = Lo.getValueType();
    Vec = DAG.getNode(BaseOpc, DL, VecEVT, Lo, Hi);
  }

  // TODO: The type may need to be widened rather than split. Or widened before
  // it can be split.
  if (!isTypeLegal(VecEVT))
    return SDValue();

  MVT VecVT = VecEVT.getSimpleVT();
  MVT VecEltVT = VecVT.getVectorElementType();
  unsigned PRVOpcode = getPRVReductionOp(Op.getOpcode());

  MVT ContainerVT = VecVT;
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  }

  MVT M1VT = getLMUL1VT(ContainerVT);

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  // FIXME: This is a VLMAX splat which might be too large and can prevent
  // vsetvli removal.
  SDValue NeutralElem =
      DAG.getNeutralElement(BaseOpc, DL, VecEltVT, SDNodeFlags());
  SDValue IdentitySplat = DAG.getSplatVector(M1VT, DL, NeutralElem);
  SDValue Reduction =
      DAG.getNode(PRVOpcode, DL, M1VT, Vec, IdentitySplat, Mask, VL);
  SDValue Elt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VecEltVT, Reduction,
                             DAG.getConstant(0, DL, Subtarget.getXLenVT()));
  return DAG.getSExtOrTrunc(Elt0, DL, Op.getValueType());
}

// Given a reduction op, this function returns the matching reduction opcode,
// the vector SDValue and the scalar SDValue required to lower this to a
// PrimateISD node.
static std::tuple<unsigned, SDValue, SDValue>
getPRVFPReductionOpAndOperands(SDValue Op, SelectionDAG &DAG, EVT EltVT) {
  SDLoc DL(Op);
  auto Flags = Op->getFlags();
  unsigned Opcode = Op.getOpcode();
  unsigned BaseOpcode = ISD::getVecReduceBaseOpcode(Opcode);
  switch (Opcode) {
  default:
    llvm_unreachable("Unhandled reduction");
  // case ISD::VECREDUCE_FADD:
  //   return std::make_tuple(PrimateISD::VECREDUCE_FADD_VL, Op.getOperand(0),
  //                          DAG.getNeutralElement(BaseOpcode, DL, EltVT, Flags));
  // case ISD::VECREDUCE_SEQ_FADD:
  //   return std::make_tuple(PrimateISD::VECREDUCE_SEQ_FADD_VL, Op.getOperand(1),
  //                          Op.getOperand(0));
  // case ISD::VECREDUCE_FMIN:
  //   return std::make_tuple(PrimateISD::VECREDUCE_FMIN_VL, Op.getOperand(0),
  //                          DAG.getNeutralElement(BaseOpcode, DL, EltVT, Flags));
  // case ISD::VECREDUCE_FMAX:
  //   return std::make_tuple(PrimateISD::VECREDUCE_FMAX_VL, Op.getOperand(0),
  //                          DAG.getNeutralElement(BaseOpcode, DL, EltVT, Flags));
  }
}

SDValue PrimateTargetLowering::lowerFPVECREDUCE(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VecEltVT = Op.getSimpleValueType();

  unsigned PRVOpcode;
  SDValue VectorVal, ScalarVal;
  std::tie(PRVOpcode, VectorVal, ScalarVal) =
      getPRVFPReductionOpAndOperands(Op, DAG, VecEltVT);
  MVT VecVT = VectorVal.getSimpleValueType();

  MVT ContainerVT = VecVT;
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    VectorVal = convertToScalableVector(ContainerVT, VectorVal, DAG, Subtarget);
  }

  MVT M1VT = getLMUL1VT(VectorVal.getSimpleValueType());

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  // FIXME: This is a VLMAX splat which might be too large and can prevent
  // vsetvli removal.
  SDValue ScalarSplat = DAG.getSplatVector(M1VT, DL, ScalarVal);
  SDValue Reduction =
      DAG.getNode(PRVOpcode, DL, M1VT, VectorVal, ScalarSplat, Mask, VL);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VecEltVT, Reduction,
                     DAG.getConstant(0, DL, Subtarget.getXLenVT()));
}

SDValue PrimateTargetLowering::lowerINSERT_SUBVECTOR(SDValue Op,
                                                   SelectionDAG &DAG) const {
  llvm_unreachable("Primate lower Insert subvector");
  // SDValue Vec = Op.getOperand(0);
  // SDValue SubVec = Op.getOperand(1);
  // MVT VecVT = Vec.getSimpleValueType();
  // MVT SubVecVT = SubVec.getSimpleValueType();

  // SDLoc DL(Op);
  // MVT XLenVT = Subtarget.getXLenVT();
  // unsigned OrigIdx = Op.getConstantOperandVal(2);
  // const PrimateRegisterInfo *TRI = Subtarget.getRegisterInfo();

  // // We don't have the ability to slide mask vectors up indexed by their i1
  // // elements; the smallest we can do is i8. Often we are able to bitcast to
  // // equivalent i8 vectors. Note that when inserting a fixed-length vector
  // // into a scalable one, we might not necessarily have enough scalable
  // // elements to safely divide by 8: nxv1i1 = insert nxv1i1, v4i1 is valid.
  // if (SubVecVT.getVectorElementType() == MVT::i1 &&
  //     (OrigIdx != 0 || !Vec.isUndef())) {
  //   if (VecVT.getVectorMinNumElements() >= 8 &&
  //       SubVecVT.getVectorMinNumElements() >= 8) {
  //     assert(OrigIdx % 8 == 0 && "Invalid index");
  //     assert(VecVT.getVectorMinNumElements() % 8 == 0 &&
  //            SubVecVT.getVectorMinNumElements() % 8 == 0 &&
  //            "Unexpected mask vector lowering");
  //     OrigIdx /= 8;
  //     SubVecVT =
  //         MVT::getVectorVT(MVT::i8, SubVecVT.getVectorMinNumElements() / 8,
  //                          SubVecVT.isScalableVector());
  //     VecVT = MVT::getVectorVT(MVT::i8, VecVT.getVectorMinNumElements() / 8,
  //                              VecVT.isScalableVector());
  //     Vec = DAG.getBitcast(VecVT, Vec);
  //     SubVec = DAG.getBitcast(SubVecVT, SubVec);
  //   } else {
  //     // We can't slide this mask vector up indexed by its i1 elements.
  //     // This poses a problem when we wish to insert a scalable vector which
  //     // can't be re-expressed as a larger type. Just choose the slow path and
  //     // extend to a larger type, then truncate back down.
  //     MVT ExtVecVT = VecVT.changeVectorElementType(MVT::i8);
  //     MVT ExtSubVecVT = SubVecVT.changeVectorElementType(MVT::i8);
  //     Vec = DAG.getNode(ISD::ZERO_EXTEND, DL, ExtVecVT, Vec);
  //     SubVec = DAG.getNode(ISD::ZERO_EXTEND, DL, ExtSubVecVT, SubVec);
  //     Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, ExtVecVT, Vec, SubVec,
  //                       Op.getOperand(2));
  //     SDValue SplatZero = DAG.getConstant(0, DL, ExtVecVT);
  //     return DAG.getSetCC(DL, VecVT, Vec, SplatZero, ISD::SETNE);
  //   }
  // }

  // // If the subvector vector is a fixed-length type, we cannot use subregister
  // // manipulation to simplify the codegen; we don't know which register of a
  // // LMUL group contains the specific subvector as we only know the minimum
  // // register size. Therefore we must slide the vector group up the full
  // // amount.
  // if (SubVecVT.isFixedLengthVector()) {
  //   if (OrigIdx == 0 && Vec.isUndef())
  //     return Op;
  //   MVT ContainerVT = VecVT;
  //   if (VecVT.isFixedLengthVector()) {
  //     ContainerVT = getContainerForFixedLengthVector(VecVT);
  //     Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  //   }
  //   SubVec = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, ContainerVT,
  //                        DAG.getUNDEF(ContainerVT), SubVec,
  //                        DAG.getConstant(0, DL, XLenVT));
  //   SDValue Mask =
  //       getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget).first;
  //   // Set the vector length to only the number of elements we care about. Note
  //   // that for slideup this includes the offset.
  //   SDValue VL =
  //       DAG.getConstant(OrigIdx + SubVecVT.getVectorNumElements(), DL, XLenVT);
  //   SDValue SlideupAmt = DAG.getConstant(OrigIdx, DL, XLenVT);
  //   SDValue Slideup = DAG.getNode(PrimateISD::VSLIDEUP_VL, DL, ContainerVT, Vec,
  //                                 SubVec, SlideupAmt, Mask, VL);
  //   if (VecVT.isFixedLengthVector())
  //     Slideup = convertFromScalableVector(VecVT, Slideup, DAG, Subtarget);
  //   return DAG.getBitcast(Op.getValueType(), Slideup);
  // }

  // unsigned SubRegIdx, RemIdx;
  // std::tie(SubRegIdx, RemIdx) =
  //     PrimateTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
  //         VecVT, SubVecVT, OrigIdx, TRI);

  // PrimateII::VLMUL SubVecLMUL = PrimateTargetLowering::getLMUL(SubVecVT);
  // bool IsSubVecPartReg = SubVecLMUL == PrimateII::VLMUL::LMUL_F2 ||
  //                        SubVecLMUL == PrimateII::VLMUL::LMUL_F4 ||
  //                        SubVecLMUL == PrimateII::VLMUL::LMUL_F8;

  // // 1. If the Idx has been completely eliminated and this subvector's size is
  // // a vector register or a multiple thereof, or the surrounding elements are
  // // undef, then this is a subvector insert which naturally aligns to a vector
  // // register. These can easily be handled using subregister manipulation.
  // // 2. If the subvector is smaller than a vector register, then the insertion
  // // must preserve the undisturbed elements of the register. We do this by
  // // lowering to an EXTRACT_SUBVECTOR grabbing the nearest LMUL=1 vector type
  // // (which resolves to a subregister copy), performing a VSLIDEUP to place the
  // // subvector within the vector register, and an INSERT_SUBVECTOR of that
  // // LMUL=1 type back into the larger vector (resolving to another subregister
  // // operation). See below for how our VSLIDEUP works. We go via a LMUL=1 type
  // // to avoid allocating a large register group to hold our subvector.
  // if (RemIdx == 0 && (!IsSubVecPartReg || Vec.isUndef()))
  //   return Op;

  // // VSLIDEUP works by leaving elements 0<i<OFFSET undisturbed, elements
  // // OFFSET<=i<VL set to the "subvector" and vl<=i<VLMAX set to the tail policy
  // // (in our case undisturbed). This means we can set up a subvector insertion
  // // where OFFSET is the insertion offset, and the VL is the OFFSET plus the
  // // size of the subvector.
  // MVT InterSubVT = VecVT;
  // SDValue AlignedExtract = Vec;
  // unsigned AlignedIdx = OrigIdx - RemIdx;
  // if (VecVT.bitsGT(getLMUL1VT(VecVT))) {
  //   InterSubVT = getLMUL1VT(VecVT);
  //   // Extract a subvector equal to the nearest full vector register type. This
  //   // should resolve to a EXTRACT_SUBREG instruction.
  //   AlignedExtract = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, InterSubVT, Vec,
  //                                DAG.getConstant(AlignedIdx, DL, XLenVT));
  // }

  // SDValue SlideupAmt = DAG.getConstant(RemIdx, DL, XLenVT);
  // // For scalable vectors this must be further multiplied by vscale.
  // SlideupAmt = DAG.getNode(ISD::VSCALE, DL, XLenVT, SlideupAmt);

  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultScalableVLOps(VecVT, DL, DAG, Subtarget);

  // // Construct the vector length corresponding to RemIdx + length(SubVecVT).
  // VL = DAG.getConstant(SubVecVT.getVectorMinNumElements(), DL, XLenVT);
  // VL = DAG.getNode(ISD::VSCALE, DL, XLenVT, VL);
  // VL = DAG.getNode(ISD::ADD, DL, XLenVT, SlideupAmt, VL);

  // SubVec = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, InterSubVT,
  //                      DAG.getUNDEF(InterSubVT), SubVec,
  //                      DAG.getConstant(0, DL, XLenVT));

  // SDValue Slideup = DAG.getNode(PrimateISD::VSLIDEUP_VL, DL, InterSubVT,
  //                               AlignedExtract, SubVec, SlideupAmt, Mask, VL);

  // // If required, insert this subvector back into the correct vector register.
  // // This should resolve to an INSERT_SUBREG instruction.
  // if (VecVT.bitsGT(InterSubVT))
  //   Slideup = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VecVT, Vec, Slideup,
  //                         DAG.getConstant(AlignedIdx, DL, XLenVT));

  // // We might have bitcast from a mask type: cast back to the original type if
  // // required.
  // return DAG.getBitcast(Op.getSimpleValueType(), Slideup);
}

SDValue PrimateTargetLowering::lowerEXTRACT_SUBVECTOR(SDValue Op,
                                                    SelectionDAG &DAG) const {
  llvm_unreachable("Primate lower EXTRACT_SUBVECTOR");
  // SDValue Vec = Op.getOperand(0);
  // MVT SubVecVT = Op.getSimpleValueType();
  // MVT VecVT = Vec.getSimpleValueType();

  // SDLoc DL(Op);
  // MVT XLenVT = Subtarget.getXLenVT();
  // unsigned OrigIdx = Op.getConstantOperandVal(1);
  // const PrimateRegisterInfo *TRI = Subtarget.getRegisterInfo();

  // // We don't have the ability to slide mask vectors down indexed by their i1
  // // elements; the smallest we can do is i8. Often we are able to bitcast to
  // // equivalent i8 vectors. Note that when extracting a fixed-length vector
  // // from a scalable one, we might not necessarily have enough scalable
  // // elements to safely divide by 8: v8i1 = extract nxv1i1 is valid.
  // if (SubVecVT.getVectorElementType() == MVT::i1 && OrigIdx != 0) {
  //   if (VecVT.getVectorMinNumElements() >= 8 &&
  //       SubVecVT.getVectorMinNumElements() >= 8) {
  //     assert(OrigIdx % 8 == 0 && "Invalid index");
  //     assert(VecVT.getVectorMinNumElements() % 8 == 0 &&
  //            SubVecVT.getVectorMinNumElements() % 8 == 0 &&
  //            "Unexpected mask vector lowering");
  //     OrigIdx /= 8;
  //     SubVecVT =
  //         MVT::getVectorVT(MVT::i8, SubVecVT.getVectorMinNumElements() / 8,
  //                          SubVecVT.isScalableVector());
  //     VecVT = MVT::getVectorVT(MVT::i8, VecVT.getVectorMinNumElements() / 8,
  //                              VecVT.isScalableVector());
  //     Vec = DAG.getBitcast(VecVT, Vec);
  //   } else {
  //     // We can't slide this mask vector down, indexed by its i1 elements.
  //     // This poses a problem when we wish to extract a scalable vector which
  //     // can't be re-expressed as a larger type. Just choose the slow path and
  //     // extend to a larger type, then truncate back down.
  //     // TODO: We could probably improve this when extracting certain fixed
  //     // from fixed, where we can extract as i8 and shift the correct element
  //     // right to reach the desired subvector?
  //     MVT ExtVecVT = VecVT.changeVectorElementType(MVT::i8);
  //     MVT ExtSubVecVT = SubVecVT.changeVectorElementType(MVT::i8);
  //     Vec = DAG.getNode(ISD::ZERO_EXTEND, DL, ExtVecVT, Vec);
  //     Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, ExtSubVecVT, Vec,
  //                       Op.getOperand(1));
  //     SDValue SplatZero = DAG.getConstant(0, DL, ExtSubVecVT);
  //     return DAG.getSetCC(DL, SubVecVT, Vec, SplatZero, ISD::SETNE);
  //   }
  // }

  // // If the subvector vector is a fixed-length type, we cannot use subregister
  // // manipulation to simplify the codegen; we don't know which register of a
  // // LMUL group contains the specific subvector as we only know the minimum
  // // register size. Therefore we must slide the vector group down the full
  // // amount.
  // if (SubVecVT.isFixedLengthVector()) {
  //   // With an index of 0 this is a cast-like subvector, which can be performed
  //   // with subregister operations.
  //   if (OrigIdx == 0)
  //     return Op;
  //   MVT ContainerVT = VecVT;
  //   if (VecVT.isFixedLengthVector()) {
  //     ContainerVT = getContainerForFixedLengthVector(VecVT);
  //     Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  //   }
  //   SDValue Mask =
  //       getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget).first;
  //   // Set the vector length to only the number of elements we care about. This
  //   // avoids sliding down elements we're going to discard straight away.
  //   SDValue VL = DAG.getConstant(SubVecVT.getVectorNumElements(), DL, XLenVT);
  //   SDValue SlidedownAmt = DAG.getConstant(OrigIdx, DL, XLenVT);
  //   SDValue Slidedown =
  //       DAG.getNode(PrimateISD::VSLIDEDOWN_VL, DL, ContainerVT,
  //                   DAG.getUNDEF(ContainerVT), Vec, SlidedownAmt, Mask, VL);
  //   // Now we can use a cast-like subvector extract to get the result.
  //   Slidedown = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVecVT, Slidedown,
  //                           DAG.getConstant(0, DL, XLenVT));
  //   return DAG.getBitcast(Op.getValueType(), Slidedown);
  // }

  // unsigned SubRegIdx, RemIdx;
  // std::tie(SubRegIdx, RemIdx) =
  //     PrimateTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
  //         VecVT, SubVecVT, OrigIdx, TRI);

  // // If the Idx has been completely eliminated then this is a subvector extract
  // // which naturally aligns to a vector register. These can easily be handled
  // // using subregister manipulation.
  // if (RemIdx == 0)
  //   return Op;

  // // Else we must shift our vector register directly to extract the subvector.
  // // Do this using VSLIDEDOWN.

  // // If the vector type is an LMUL-group type, extract a subvector equal to the
  // // nearest full vector register type. This should resolve to a EXTRACT_SUBREG
  // // instruction.
  // MVT InterSubVT = VecVT;
  // if (VecVT.bitsGT(getLMUL1VT(VecVT))) {
  //   InterSubVT = getLMUL1VT(VecVT);
  //   Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, InterSubVT, Vec,
  //                     DAG.getConstant(OrigIdx - RemIdx, DL, XLenVT));
  // }

  // // Slide this vector register down by the desired number of elements in order
  // // to place the desired subvector starting at element 0.
  // SDValue SlidedownAmt = DAG.getConstant(RemIdx, DL, XLenVT);
  // // For scalable vectors this must be further multiplied by vscale.
  // SlidedownAmt = DAG.getNode(ISD::VSCALE, DL, XLenVT, SlidedownAmt);

  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultScalableVLOps(InterSubVT, DL, DAG, Subtarget);
  // SDValue Slidedown =
  //     DAG.getNode(PrimateISD::VSLIDEDOWN_VL, DL, InterSubVT,
  //                 DAG.getUNDEF(InterSubVT), Vec, SlidedownAmt, Mask, VL);

  // // Now the vector is in the right position, extract our final subvector. This
  // // should resolve to a COPY.
  // Slidedown = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVecVT, Slidedown,
  //                         DAG.getConstant(0, DL, XLenVT));

  // // We might have bitcast from a mask type: cast back to the original type if
  // // required.
  // return DAG.getBitcast(Op.getSimpleValueType(), Slidedown);
}

// Lower step_vector to the vid instruction. Any non-identity step value must
// be accounted for my manual expansion.
SDValue PrimateTargetLowering::lowerSTEP_VECTOR(SDValue Op,
                                              SelectionDAG &DAG) const {
  llvm_unreachable("primate lower STEP_VECTOR");
  // SDLoc DL(Op);
  // MVT VT = Op.getSimpleValueType();
  // MVT XLenVT = Subtarget.getXLenVT();
  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultScalableVLOps(VT, DL, DAG, Subtarget);
  // SDValue StepVec = DAG.getNode(PrimateISD::VID_VL, DL, VT, Mask, VL);
  // uint64_t StepValImm = Op.getConstantOperandVal(0);
  // if (StepValImm != 1) {
  //   if (isPowerOf2_64(StepValImm)) {
  //     SDValue StepVal =
  //         DAG.getNode(PrimateISD::VMV_V_X_VL, DL, VT,
  //                     DAG.getConstant(Log2_64(StepValImm), DL, XLenVT));
  //     StepVec = DAG.getNode(ISD::SHL, DL, VT, StepVec, StepVal);
  //   } else {
  //     SDValue StepVal = lowerScalarSplat(
  //         DAG.getConstant(StepValImm, DL, VT.getVectorElementType()), VL, VT,
  //         DL, DAG, Subtarget);
  //     StepVec = DAG.getNode(ISD::MUL, DL, VT, StepVec, StepVal);
  //   }
  // }
  // return StepVec;
}

// Implement vector_reverse using vrgather.vv with indices determined by
// subtracting the id of each element from (VLMAX-1). This will convert
// the indices like so:
// (0, 1,..., VLMAX-2, VLMAX-1) -> (VLMAX-1, VLMAX-2,..., 1, 0).
// TODO: This code assumes VLMAX <= 65536 for LMUL=8 SEW=16.
SDValue PrimateTargetLowering::lowerVECTOR_REVERSE(SDValue Op,
                                                 SelectionDAG &DAG) const {
  llvm_unreachable("Primate lower VECTOR_REVERSE");
  // SDLoc DL(Op);
  // MVT VecVT = Op.getSimpleValueType();
  // unsigned EltSize = VecVT.getScalarSizeInBits();
  // unsigned MinSize = VecVT.getSizeInBits().getKnownMinValue();

  // unsigned MaxVLMAX = 0;
  // unsigned VectorBitsMax = Subtarget.getMaxPRVVectorSizeInBits();
  // if (VectorBitsMax != 0)
  //   MaxVLMAX = ((VectorBitsMax / EltSize) * MinSize) / Primate::PRVBitsPerBlock;

  // unsigned GatherOpc = PrimateISD::VRGATHER_VV_VL;
  // MVT IntVT = VecVT.changeVectorElementTypeToInteger();

  // // If this is SEW=8 and VLMAX is unknown or more than 256, we need
  // // to use vrgatherei16.vv.
  // // TODO: It's also possible to use vrgatherei16.vv for other types to
  // // decrease register width for the index calculation.
  // if ((MaxVLMAX == 0 || MaxVLMAX > 256) && EltSize == 8) {
  //   // If this is LMUL=8, we have to split before can use vrgatherei16.vv.
  //   // Reverse each half, then reassemble them in reverse order.
  //   // NOTE: It's also possible that after splitting that VLMAX no longer
  //   // requires vrgatherei16.vv.
  //   if (MinSize == (8 * Primate::PRVBitsPerBlock)) {
  //     SDValue Lo, Hi;
  //     std::tie(Lo, Hi) = DAG.SplitVectorOperand(Op.getNode(), 0);
  //     EVT LoVT, HiVT;
  //     std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(VecVT);
  //     Lo = DAG.getNode(ISD::VECTOR_REVERSE, DL, LoVT, Lo);
  //     Hi = DAG.getNode(ISD::VECTOR_REVERSE, DL, HiVT, Hi);
  //     // Reassemble the low and high pieces reversed.
  //     // FIXME: This is a CONCAT_VECTORS.
  //     SDValue Res =
  //         DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VecVT, DAG.getUNDEF(VecVT), Hi,
  //                     DAG.getIntPtrConstant(0, DL));
  //     return DAG.getNode(
  //         ISD::INSERT_SUBVECTOR, DL, VecVT, Res, Lo,
  //         DAG.getIntPtrConstant(LoVT.getVectorMinNumElements(), DL));
  //   }

  //   // Just promote the int type to i16 which will double the LMUL.
  //   IntVT = MVT::getVectorVT(MVT::i16, VecVT.getVectorElementCount());
  //   GatherOpc = PrimateISD::VRGATHEREI16_VV_VL;
  // }

  // MVT XLenVT = Subtarget.getXLenVT();
  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultScalableVLOps(VecVT, DL, DAG, Subtarget);

  // // Calculate VLMAX-1 for the desired SEW.
  // unsigned MinElts = VecVT.getVectorMinNumElements();
  // SDValue VLMax = DAG.getNode(ISD::VSCALE, DL, XLenVT,
  //                             DAG.getConstant(MinElts, DL, XLenVT));
  // SDValue VLMinus1 =
  //     DAG.getNode(ISD::SUB, DL, XLenVT, VLMax, DAG.getConstant(1, DL, XLenVT));

  // // Splat VLMAX-1 taking care to handle SEW==64 on PR32.
  // bool IsPR32E64 =
  //     !Subtarget.is64Bit() && IntVT.getVectorElementType() == MVT::i64;
  // SDValue SplatVL;
  // if (!IsPR32E64)
  //   SplatVL = DAG.getSplatVector(IntVT, DL, VLMinus1);
  // else
  //   SplatVL = DAG.getNode(PrimateISD::SPLAT_VECTOR_I64, DL, IntVT, VLMinus1);

  // SDValue VID = DAG.getNode(PrimateISD::VID_VL, DL, IntVT, Mask, VL);
  // SDValue Indices =
  //     DAG.getNode(PrimateISD::SUB_VL, DL, IntVT, SplatVL, VID, Mask, VL);

  // return DAG.getNode(GatherOpc, DL, VecVT, Op.getOperand(0), Indices, Mask, VL);
}

SDValue
PrimateTargetLowering::lowerFixedLengthVectorLoadToPRV(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *Load = cast<LoadSDNode>(Op);

  assert(allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                        Load->getMemoryVT(),
                                        *Load->getMemOperand()) &&
         "Expecting a correctly-aligned load");

  MVT VT = Op.getSimpleValueType();
  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  SDValue VL =
      DAG.getConstant(VT.getVectorNumElements(), DL, Subtarget.getXLenVT());

  SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});
  SDValue NewLoad = DAG.getMemIntrinsicNode(
      PrimateISD::VLE_VL, DL, VTs, {Load->getChain(), Load->getBasePtr(), VL},
      Load->getMemoryVT(), Load->getMemOperand());

  SDValue Result = convertFromScalableVector(VT, NewLoad, DAG, Subtarget);
  return DAG.getMergeValues({Result, Load->getChain()}, DL);
}

SDValue
PrimateTargetLowering::lowerFixedLengthVectorStoreToPRV(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *Store = cast<StoreSDNode>(Op);

  assert(allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                        Store->getMemoryVT(),
                                        *Store->getMemOperand()) &&
         "Expecting a correctly-aligned store");

  SDValue StoreVal = Store->getValue();
  MVT VT = StoreVal.getSimpleValueType();

  // If the size less than a byte, we need to pad with zeros to make a byte.
  if (VT.getVectorElementType() == MVT::i1 && VT.getVectorNumElements() < 8) {
    VT = MVT::v8i1;
    StoreVal = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT,
                           DAG.getConstant(0, DL, VT), StoreVal,
                           DAG.getIntPtrConstant(0, DL));
  }

  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  SDValue VL =
      DAG.getConstant(VT.getVectorNumElements(), DL, Subtarget.getXLenVT());

  SDValue NewValue =
      convertToScalableVector(ContainerVT, StoreVal, DAG, Subtarget);
  return DAG.getMemIntrinsicNode(
      PrimateISD::VSE_VL, DL, DAG.getVTList(MVT::Other),
      {Store->getChain(), NewValue, Store->getBasePtr(), VL},
      Store->getMemoryVT(), Store->getMemOperand());
}

SDValue PrimateTargetLowering::lowerMLOAD(SDValue Op, SelectionDAG &DAG) const {
  auto *Load = cast<MaskedLoadSDNode>(Op);

  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  SDValue Mask = Load->getMask();
  SDValue PassThru = Load->getPassThru();
  SDValue VL;

  MVT ContainerVT = VT;
  if (VT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VT);
    MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());

    Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
    PassThru = convertToScalableVector(ContainerVT, PassThru, DAG, Subtarget);
    VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);
  } else
    VL = DAG.getRegister(Primate::X0, XLenVT);

  SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});
  SDValue IntID = DAG.getTargetConstant(Intrinsic::primate_vle_mask, DL, XLenVT);
  SDValue Ops[] = {Load->getChain(),   IntID, PassThru,
                   Load->getBasePtr(), Mask,  VL};
  SDValue Result =
      DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops,
                              Load->getMemoryVT(), Load->getMemOperand());
  SDValue Chain = Result.getValue(1);

  if (VT.isFixedLengthVector())
    Result = convertFromScalableVector(VT, Result, DAG, Subtarget);

  return DAG.getMergeValues({Result, Chain}, DL);
}

SDValue PrimateTargetLowering::lowerMSTORE(SDValue Op, SelectionDAG &DAG) const {
  auto *Store = cast<MaskedStoreSDNode>(Op);

  SDLoc DL(Op);
  SDValue Val = Store->getValue();
  SDValue Mask = Store->getMask();
  MVT VT = Val.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();
  SDValue VL;

  MVT ContainerVT = VT;
  if (VT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VT);
    MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());

    Val = convertToScalableVector(ContainerVT, Val, DAG, Subtarget);
    Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
    VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);
  } else
    VL = DAG.getRegister(Primate::X0, XLenVT);

  SDValue IntID = DAG.getTargetConstant(Intrinsic::primate_vse_mask, DL, XLenVT);
  return DAG.getMemIntrinsicNode(
      ISD::INTRINSIC_VOID, DL, DAG.getVTList(MVT::Other),
      {Store->getChain(), IntID, Val, Store->getBasePtr(), Mask, VL},
      Store->getMemoryVT(), Store->getMemOperand());
}

SDValue
PrimateTargetLowering::lowerFixedLengthVectorSetccToPRV(SDValue Op,
                                                      SelectionDAG &DAG) const {
  llvm_unreachable("Primate lower FixedLengthVectorSetccToPRV");
  // MVT InVT = Op.getOperand(0).getSimpleValueType();
  // MVT ContainerVT = getContainerForFixedLengthVector(InVT);

  // MVT VT = Op.getSimpleValueType();

  // SDValue Op1 =
  //     convertToScalableVector(ContainerVT, Op.getOperand(0), DAG, Subtarget);
  // SDValue Op2 =
  //     convertToScalableVector(ContainerVT, Op.getOperand(1), DAG, Subtarget);

  // SDLoc DL(Op);
  // SDValue VL =
  //     DAG.getConstant(VT.getVectorNumElements(), DL, Subtarget.getXLenVT());

  // MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
  // SDValue Mask = DAG.getNode(PrimateISD::VMSET_VL, DL, MaskVT, VL);

  // SDValue Cmp = DAG.getNode(PrimateISD::SETCC_VL, DL, MaskVT, Op1, Op2,
  //                           Op.getOperand(2), Mask, VL);

  // return convertFromScalableVector(VT, Cmp, DAG, Subtarget);
}

SDValue PrimateTargetLowering::lowerFixedLengthVectorLogicOpToPRV(
    SDValue Op, SelectionDAG &DAG, unsigned MaskOpc, unsigned VecOpc) const {
  MVT VT = Op.getSimpleValueType();

  if (VT.getVectorElementType() == MVT::i1)
    return lowerToScalableOp(Op, DAG, MaskOpc, /*HasMask*/ false);

  return lowerToScalableOp(Op, DAG, VecOpc, /*HasMask*/ true);
}

SDValue
PrimateTargetLowering::lowerFixedLengthVectorShiftToPRV(SDValue Op,
                                                      SelectionDAG &DAG) const {
  unsigned Opc;
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Unexpected opcode!");
  // case ISD::SHL: Opc = PrimateISD::SHL_VL; break;
  // case ISD::SRA: Opc = PrimateISD::SRA_VL; break;
  // case ISD::SRL: Opc = PrimateISD::SRL_VL; break;
  }

  return lowerToScalableOp(Op, DAG, Opc);
}

// Lower vector ABS to smax(X, sub(0, X)).
SDValue PrimateTargetLowering::lowerABS(SDValue Op, SelectionDAG &DAG) const {
  llvm_unreachable("Primate lower ABS");
  // SDLoc DL(Op);
  // MVT VT = Op.getSimpleValueType();
  // SDValue X = Op.getOperand(0);

  // assert(VT.isFixedLengthVector() && "Unexpected type");

  // MVT ContainerVT = getContainerForFixedLengthVector(VT);
  // X = convertToScalableVector(ContainerVT, X, DAG, Subtarget);

  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  // SDValue SplatZero =
  //     DAG.getNode(PrimateISD::VMV_V_X_VL, DL, ContainerVT,
  //                 DAG.getConstant(0, DL, Subtarget.getXLenVT()));
  // SDValue NegX =
  //     DAG.getNode(PrimateISD::SUB_VL, DL, ContainerVT, SplatZero, X, Mask, VL);
  // SDValue Max =
  //     DAG.getNode(PrimateISD::SMAX_VL, DL, ContainerVT, X, NegX, Mask, VL);

  // return convertFromScalableVector(VT, Max, DAG, Subtarget);
}

SDValue PrimateTargetLowering::lowerFixedLengthVectorFCOPYSIGNToPRV(
    SDValue Op, SelectionDAG &DAG) const {
  llvm_unreachable("Primate lower FixedLengthVectorFCOPYSIGNToPRV");
  // SDLoc DL(Op);
  // MVT VT = Op.getSimpleValueType();
  // SDValue Mag = Op.getOperand(0);
  // SDValue Sign = Op.getOperand(1);
  // assert(Mag.getValueType() == Sign.getValueType() &&
  //        "Can only handle COPYSIGN with matching types.");

  // MVT ContainerVT = getContainerForFixedLengthVector(VT);
  // Mag = convertToScalableVector(ContainerVT, Mag, DAG, Subtarget);
  // Sign = convertToScalableVector(ContainerVT, Sign, DAG, Subtarget);

  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  // SDValue CopySign =
  //     DAG.getNode(PrimateISD::FCOPYSIGN_VL, DL, ContainerVT, Mag, Sign, Mask, VL);

  // return convertFromScalableVector(VT, CopySign, DAG, Subtarget);
}

SDValue PrimateTargetLowering::lowerFixedLengthVectorSelectToPRV(
    SDValue Op, SelectionDAG &DAG) const {
  llvm_unreachable("primate lower FixedLengthVectorSelectToPRV");
  // MVT VT = Op.getSimpleValueType();
  // MVT ContainerVT = getContainerForFixedLengthVector(VT);

  // MVT I1ContainerVT =
  //     MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());

  // SDValue CC =
  //     convertToScalableVector(I1ContainerVT, Op.getOperand(0), DAG, Subtarget);
  // SDValue Op1 =
  //     convertToScalableVector(ContainerVT, Op.getOperand(1), DAG, Subtarget);
  // SDValue Op2 =
  //     convertToScalableVector(ContainerVT, Op.getOperand(2), DAG, Subtarget);

  // SDLoc DL(Op);
  // SDValue Mask, VL;
  // std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  // SDValue Select =
  //     DAG.getNode(PrimateISD::VSELECT_VL, DL, ContainerVT, CC, Op1, Op2, VL);

  // return convertFromScalableVector(VT, Select, DAG, Subtarget);
}

SDValue PrimateTargetLowering::lowerToScalableOp(SDValue Op, SelectionDAG &DAG,
                                               unsigned NewOpc,
                                               bool HasMask) const {
  MVT VT = Op.getSimpleValueType();
  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  // Create list of operands by converting existing ones to scalable types.
  SmallVector<SDValue, 6> Ops;
  for (const SDValue &V : Op->op_values()) {
    assert(!isa<VTSDNode>(V) && "Unexpected VTSDNode node!");

    // Pass through non-vector operands.
    if (!V.getValueType().isVector()) {
      Ops.push_back(V);
      continue;
    }

    // "cast" fixed length vector to a scalable vector.
    assert(usePRVForFixedLengthVectorVT(V.getSimpleValueType()) &&
           "Only fixed length vectors are supported!");
    Ops.push_back(convertToScalableVector(ContainerVT, V, DAG, Subtarget));
  }

  SDLoc DL(Op);
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);
  if (HasMask)
    Ops.push_back(Mask);
  Ops.push_back(VL);

  SDValue ScalableRes = DAG.getNode(NewOpc, DL, ContainerVT, Ops);
  return convertFromScalableVector(VT, ScalableRes, DAG, Subtarget);
}

// Lower a VP_* ISD node to the corresponding PrimateISD::*_VL node:
// * Operands of each node are assumed to be in the same order.
// * The EVL operand is promoted from i32 to i64 on PR64.
// * Fixed-length vectors are converted to their scalable-vector container
//   types.
SDValue PrimateTargetLowering::lowerVPOp(SDValue Op, SelectionDAG &DAG,
                                       unsigned PrimateISDOpc) const {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  SmallVector<SDValue, 4> Ops;

  for (const auto &OpIdx : enumerate(Op->ops())) {
    SDValue V = OpIdx.value();
    assert(!isa<VTSDNode>(V) && "Unexpected VTSDNode node!");
    // Pass through operands which aren't fixed-length vectors.
    if (!V.getValueType().isFixedLengthVector()) {
      Ops.push_back(V);
      continue;
    }
    // "cast" fixed length vector to a scalable vector.
    MVT OpVT = V.getSimpleValueType();
    MVT ContainerVT = getContainerForFixedLengthVector(OpVT);
    assert(usePRVForFixedLengthVectorVT(OpVT) &&
           "Only fixed length vectors are supported!");
    Ops.push_back(convertToScalableVector(ContainerVT, V, DAG, Subtarget));
  }

  if (!VT.isFixedLengthVector())
    return DAG.getNode(PrimateISDOpc, DL, VT, Ops);

  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  SDValue VPOp = DAG.getNode(PrimateISDOpc, DL, ContainerVT, Ops);

  return convertFromScalableVector(VT, VPOp, DAG, Subtarget);
}

// Custom lower MGATHER to a legalized form for PRV. It will then be matched to
// a PRV indexed load. The PRV indexed load instructions only support the
// "unsigned unscaled" addressing mode; indices are implicitly zero-extended or
// truncated to XLEN and are treated as byte offsets. Any signed or scaled
// indexing is extended to the XLEN value type and scaled accordingly.
SDValue PrimateTargetLowering::lowerMGATHER(SDValue Op, SelectionDAG &DAG) const {
  auto *MGN = cast<MaskedGatherSDNode>(Op.getNode());
  SDLoc DL(Op);

  SDValue Index = MGN->getIndex();
  SDValue Mask = MGN->getMask();
  SDValue PassThru = MGN->getPassThru();

  MVT VT = Op.getSimpleValueType();
  MVT IndexVT = Index.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Unexpected VTs!");
  assert(MGN->getBasePtr().getSimpleValueType() == XLenVT &&
         "Unexpected pointer type");
  // Targets have to explicitly opt-in for extending vector loads.
  assert(MGN->getExtensionType() == ISD::NON_EXTLOAD &&
         "Unexpected extending MGATHER");

  // If the mask is known to be all ones, optimize to an unmasked intrinsic;
  // the selection of the masked intrinsics doesn't do this for us.
  bool IsUnmasked = ISD::isConstantSplatVectorAllOnes(Mask.getNode());

  SDValue VL;
  MVT ContainerVT = VT;
  if (VT.isFixedLengthVector()) {
    // We need to use the larger of the result and index type to determine the
    // scalable type to use so we don't increase LMUL for any operand/result.
    if (VT.bitsGE(IndexVT)) {
      ContainerVT = getContainerForFixedLengthVector(VT);
      IndexVT = MVT::getVectorVT(IndexVT.getVectorElementType(),
                                 ContainerVT.getVectorElementCount());
    } else {
      IndexVT = getContainerForFixedLengthVector(IndexVT);
      ContainerVT = MVT::getVectorVT(ContainerVT.getVectorElementType(),
                                     IndexVT.getVectorElementCount());
    }

    Index = convertToScalableVector(IndexVT, Index, DAG, Subtarget);

    if (!IsUnmasked) {
      MVT MaskVT =
          MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
      Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
      PassThru = convertToScalableVector(ContainerVT, PassThru, DAG, Subtarget);
    }

    VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);
  } else
    VL = DAG.getRegister(Primate::X0, XLenVT);

  unsigned IntID =
      IsUnmasked ? Intrinsic::primate_vluxei : Intrinsic::primate_vluxei_mask;
  SmallVector<SDValue, 8> Ops{MGN->getChain(),
                              DAG.getTargetConstant(IntID, DL, XLenVT)};
  if (!IsUnmasked)
    Ops.push_back(PassThru);
  Ops.push_back(MGN->getBasePtr());
  Ops.push_back(Index);
  if (!IsUnmasked)
    Ops.push_back(Mask);
  Ops.push_back(VL);

  SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});
  SDValue Result =
      DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops,
                              MGN->getMemoryVT(), MGN->getMemOperand());
  SDValue Chain = Result.getValue(1);

  if (VT.isFixedLengthVector())
    Result = convertFromScalableVector(VT, Result, DAG, Subtarget);

  return DAG.getMergeValues({Result, Chain}, DL);
}

// Custom lower MSCATTER to a legalized form for PRV. It will then be matched to
// a PRV indexed store. The PRV indexed store instructions only support the
// "unsigned unscaled" addressing mode; indices are implicitly zero-extended or
// truncated to XLEN and are treated as byte offsets. Any signed or scaled
// indexing is extended to the XLEN value type and scaled accordingly.
SDValue PrimateTargetLowering::lowerMSCATTER(SDValue Op,
                                           SelectionDAG &DAG) const {
  auto *MSN = cast<MaskedScatterSDNode>(Op.getNode());
  SDLoc DL(Op);
  SDValue Index = MSN->getIndex();
  SDValue Mask = MSN->getMask();
  SDValue Val = MSN->getValue();

  MVT VT = Val.getSimpleValueType();
  MVT IndexVT = Index.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Unexpected VTs!");
  assert(MSN->getBasePtr().getSimpleValueType() == XLenVT &&
         "Unexpected pointer type");
  // Targets have to explicitly opt-in for extending vector loads and
  // truncating vector stores.
  assert(!MSN->isTruncatingStore() && "Unexpected extending MSCATTER");

  // If the mask is known to be all ones, optimize to an unmasked intrinsic;
  // the selection of the masked intrinsics doesn't do this for us.
  bool IsUnmasked = ISD::isConstantSplatVectorAllOnes(Mask.getNode());

  SDValue VL;
  if (VT.isFixedLengthVector()) {
    // We need to use the larger of the value and index type to determine the
    // scalable type to use so we don't increase LMUL for any operand/result.
    MVT ContainerVT;
    if (VT.bitsGE(IndexVT)) {
      ContainerVT = getContainerForFixedLengthVector(VT);
      IndexVT = MVT::getVectorVT(IndexVT.getVectorElementType(),
                                 ContainerVT.getVectorElementCount());
    } else {
      IndexVT = getContainerForFixedLengthVector(IndexVT);
      ContainerVT = MVT::getVectorVT(VT.getVectorElementType(),
                                     IndexVT.getVectorElementCount());
    }

    Index = convertToScalableVector(IndexVT, Index, DAG, Subtarget);
    Val = convertToScalableVector(ContainerVT, Val, DAG, Subtarget);

    if (!IsUnmasked) {
      MVT MaskVT =
          MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
      Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
    }

    VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);
  } else
    VL = DAG.getRegister(Primate::X0, XLenVT);

  unsigned IntID =
      IsUnmasked ? Intrinsic::primate_vsoxei : Intrinsic::primate_vsoxei_mask;
  SmallVector<SDValue, 8> Ops{MSN->getChain(),
                              DAG.getTargetConstant(IntID, DL, XLenVT)};
  Ops.push_back(Val);
  Ops.push_back(MSN->getBasePtr());
  Ops.push_back(Index);
  if (!IsUnmasked)
    Ops.push_back(Mask);
  Ops.push_back(VL);

  return DAG.getMemIntrinsicNode(ISD::INTRINSIC_VOID, DL, MSN->getVTList(), Ops,
                                 MSN->getMemoryVT(), MSN->getMemOperand());
}

SDValue PrimateTargetLowering::lowerGET_ROUNDING(SDValue Op,
                                               SelectionDAG &DAG) const {
  const MVT XLenVT = Subtarget.getXLenVT();
  SDLoc DL(Op);
  SDValue Chain = Op->getOperand(0);
  SDValue SysRegNo = DAG.getConstant(
      PrimateSysReg::lookupSysRegByName("FRM")->Encoding, DL, XLenVT);
  SDVTList VTs = DAG.getVTList(XLenVT, MVT::Other);
  SDValue RM = DAG.getNode(PrimateISD::READ_CSR, DL, VTs, Chain, SysRegNo);

  // Encoding used for rounding mode in Primate differs from that used in
  // FLT_ROUNDS. To convert it the Primate rounding mode is used as an index in a
  // table, which consists of a sequence of 4-bit fields, each representing
  // corresponding FLT_ROUNDS mode.
  static const int Table =
      (int(RoundingMode::NearestTiesToEven) << 4 * PrimateFPRndMode::RNE) |
      (int(RoundingMode::TowardZero) << 4 * PrimateFPRndMode::RTZ) |
      (int(RoundingMode::TowardNegative) << 4 * PrimateFPRndMode::RDN) |
      (int(RoundingMode::TowardPositive) << 4 * PrimateFPRndMode::RUP) |
      (int(RoundingMode::NearestTiesToAway) << 4 * PrimateFPRndMode::RMM);

  SDValue Shift =
      DAG.getNode(ISD::SHL, DL, XLenVT, RM, DAG.getConstant(2, DL, XLenVT));
  SDValue Shifted = DAG.getNode(ISD::SRL, DL, XLenVT,
                                DAG.getConstant(Table, DL, XLenVT), Shift);
  SDValue Masked = DAG.getNode(ISD::AND, DL, XLenVT, Shifted,
                               DAG.getConstant(7, DL, XLenVT));

  return DAG.getMergeValues({Masked, Chain}, DL);
}

SDValue PrimateTargetLowering::lowerSET_ROUNDING(SDValue Op,
                                               SelectionDAG &DAG) const {
  const MVT XLenVT = Subtarget.getXLenVT();
  SDLoc DL(Op);
  SDValue Chain = Op->getOperand(0);
  SDValue RMValue = Op->getOperand(1);
  SDValue SysRegNo = DAG.getConstant(
      PrimateSysReg::lookupSysRegByName("FRM")->Encoding, DL, XLenVT);

  // Encoding used for rounding mode in Primate differs from that used in
  // FLT_ROUNDS. To convert it the C rounding mode is used as an index in
  // a table, which consists of a sequence of 4-bit fields, each representing
  // corresponding Primate mode.
  static const unsigned Table =
      (PrimateFPRndMode::RNE << 4 * int(RoundingMode::NearestTiesToEven)) |
      (PrimateFPRndMode::RTZ << 4 * int(RoundingMode::TowardZero)) |
      (PrimateFPRndMode::RDN << 4 * int(RoundingMode::TowardNegative)) |
      (PrimateFPRndMode::RUP << 4 * int(RoundingMode::TowardPositive)) |
      (PrimateFPRndMode::RMM << 4 * int(RoundingMode::NearestTiesToAway));

  SDValue Shift = DAG.getNode(ISD::SHL, DL, XLenVT, RMValue,
                              DAG.getConstant(2, DL, XLenVT));
  SDValue Shifted = DAG.getNode(ISD::SRL, DL, XLenVT,
                                DAG.getConstant(Table, DL, XLenVT), Shift);
  RMValue = DAG.getNode(ISD::AND, DL, XLenVT, Shifted,
                        DAG.getConstant(0x7, DL, XLenVT));
  return DAG.getNode(PrimateISD::WRITE_CSR, DL, MVT::Other, Chain, SysRegNo,
                     RMValue);
}

// Returns the opcode of the target-specific SDNode that implements the 32-bit
// form of the given Opcode.
static PrimateISD::NodeType getPrimateWOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::SHL:
    return PrimateISD::SLLW;
  case ISD::SRA:
    return PrimateISD::SRAW;
  case ISD::SRL:
    return PrimateISD::SRLW;
  case ISD::SDIV:
    return PrimateISD::DIVW;
  case ISD::UDIV:
    return PrimateISD::DIVUW;
  case ISD::UREM:
    return PrimateISD::REMUW;
  case ISD::ROTL:
    return PrimateISD::ROLW;
  case ISD::ROTR:
    return PrimateISD::RORW;
  case PrimateISD::GREV:
    return PrimateISD::GREVW;
  case PrimateISD::GORC:
    return PrimateISD::GORCW;
  }
}

// Converts the given i8/i16/i32 operation to a target-specific SelectionDAG
// node. Because i8/i16/i32 isn't a legal type for PR64, these operations would
// otherwise be promoted to i64, making it difficult to select the
// SLLW/DIVUW/.../*W later one because the fact the operation was originally of
// type i8/i16/i32 is lost.
static SDValue customLegalizeToWOp(SDNode *N, SelectionDAG &DAG,
                                   unsigned ExtOpc = ISD::ANY_EXTEND) {
  SDLoc DL(N);
  PrimateISD::NodeType WOpcode = getPrimateWOpcode(N->getOpcode());
  SDValue NewOp0 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(1));
  SDValue NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp0, NewOp1);
  // ReplaceNodeResults requires we maintain the same type for the return value.
  return DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), NewRes);
}

// Converts the given 32-bit operation to a i64 operation with signed extension
// semantic to reduce the signed extension instructions.
static SDValue customLegalizeToWOpWithSExt(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
  SDValue NewWOp = DAG.getNode(N->getOpcode(), DL, MVT::i64, NewOp0, NewOp1);
  SDValue NewRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, NewWOp,
                               DAG.getValueType(MVT::i32));
  return DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes);
}

void PrimateTargetLowering::ReplaceNodeResults(SDNode *N,
                                             SmallVectorImpl<SDValue> &Results,
                                             SelectionDAG &DAG) const {
  SDLoc DL(N);
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom type legalize this operation!");
  case ISD::STRICT_FP_TO_SINT:
  case ISD::STRICT_FP_TO_UINT:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    bool IsStrict = N->isStrictFPOpcode();
    bool IsSigned = N->getOpcode() == ISD::FP_TO_SINT ||
                    N->getOpcode() == ISD::STRICT_FP_TO_SINT;
    SDValue Op0 = IsStrict ? N->getOperand(1) : N->getOperand(0);
    if (getTypeAction(*DAG.getContext(), Op0.getValueType()) !=
        TargetLowering::TypeSoftenFloat) {
      // FIXME: Support strict FP.
      if (IsStrict)
        return;
      if (!isTypeLegal(Op0.getValueType()))
        return;
      unsigned Opc = IsSigned ? PrimateISD::FCVT_W_PR64 : PrimateISD::FCVT_WU_PR64;
      SDValue Res = DAG.getNode(Opc, DL, MVT::i64, Op0);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      return;
    }
    // If the FP type needs to be softened, emit a library call using the 'si'
    // version. If we left it to default legalization we'd end up with 'di'. If
    // the FP type doesn't need to be softened just let generic type
    // legalization promote the result type.
    RTLIB::Libcall LC;
    if (IsSigned)
      LC = RTLIB::getFPTOSINT(Op0.getValueType(), N->getValueType(0));
    else
      LC = RTLIB::getFPTOUINT(Op0.getValueType(), N->getValueType(0));
    MakeLibCallOptions CallOptions;
    EVT OpVT = Op0.getValueType();
    CallOptions.setTypeListBeforeSoften(OpVT, N->getValueType(0), true);
    SDValue Chain = IsStrict ? N->getOperand(0) : SDValue();
    SDValue Result;
    std::tie(Result, Chain) =
        makeLibCall(DAG, LC, N->getValueType(0), Op0, CallOptions, DL, Chain);
    Results.push_back(Result);
    if (IsStrict)
      Results.push_back(Chain);
    break;
  }
  case ISD::READCYCLECOUNTER: {
    assert(!Subtarget.is64Bit() &&
           "READCYCLECOUNTER only has custom type legalization on primate32");

    SDVTList VTs = DAG.getVTList(MVT::i32, MVT::i32, MVT::Other);
    SDValue RCW =
        DAG.getNode(PrimateISD::READ_CYCLE_WIDE, DL, VTs, N->getOperand(0));

    Results.push_back(
        DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, RCW, RCW.getValue(1)));
    Results.push_back(RCW.getValue(2));
    break;
  }
  case ISD::MUL: {
    unsigned Size = N->getSimpleValueType(0).getSizeInBits();
    unsigned XLen = Subtarget.getXLen();
    // This multiply needs to be expanded, try to use MULHSU+MUL if possible.
    if (Size > XLen) {
      assert(Size == (XLen * 2) && "Unexpected custom legalisation");
      SDValue LHS = N->getOperand(0);
      SDValue RHS = N->getOperand(1);
      APInt HighMask = APInt::getHighBitsSet(Size, XLen);

      bool LHSIsU = DAG.MaskedValueIsZero(LHS, HighMask);
      bool RHSIsU = DAG.MaskedValueIsZero(RHS, HighMask);
      // We need exactly one side to be unsigned.
      if (LHSIsU == RHSIsU)
        return;

      auto MakeMULPair = [&](SDValue S, SDValue U) {
        MVT XLenVT = Subtarget.getXLenVT();
        S = DAG.getNode(ISD::TRUNCATE, DL, XLenVT, S);
        U = DAG.getNode(ISD::TRUNCATE, DL, XLenVT, U);
        SDValue Lo = DAG.getNode(ISD::MUL, DL, XLenVT, S, U);
        SDValue Hi = DAG.getNode(PrimateISD::MULHSU, DL, XLenVT, S, U);
        return DAG.getNode(ISD::BUILD_PAIR, DL, N->getValueType(0), Lo, Hi);
      };

      bool LHSIsS = DAG.ComputeNumSignBits(LHS) > XLen;
      bool RHSIsS = DAG.ComputeNumSignBits(RHS) > XLen;

      // The other operand should be signed, but still prefer MULH when
      // possible.
      if (RHSIsU && LHSIsS && !RHSIsS)
        Results.push_back(MakeMULPair(LHS, RHS));
      else if (LHSIsU && RHSIsS && !LHSIsS)
        Results.push_back(MakeMULPair(RHS, LHS));

      return;
    }
    LLVM_FALLTHROUGH;
  }
  case ISD::ADD:
  case ISD::SUB:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      return;
    Results.push_back(customLegalizeToWOpWithSExt(N, DAG));
    break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      return;
    Results.push_back(customLegalizeToWOp(N, DAG));
    break;
  case ISD::ROTL:
  case ISD::ROTR:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    Results.push_back(customLegalizeToWOp(N, DAG));
    break;
  case ISD::CTTZ:
  case ISD::CTTZ_ZERO_UNDEF:
  case ISD::CTLZ:
  case ISD::CTLZ_ZERO_UNDEF: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");

    SDValue NewOp0 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
    bool IsCTZ =
        N->getOpcode() == ISD::CTTZ || N->getOpcode() == ISD::CTTZ_ZERO_UNDEF;
    unsigned Opc = IsCTZ ? PrimateISD::CTZW : PrimateISD::CLZW;
    SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp0);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
    return;
  }
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::UREM: {
    MVT VT = N->getSimpleValueType(0);
    assert((VT == MVT::i8 || VT == MVT::i16 || VT == MVT::i32) &&
           Subtarget.is64Bit() && Subtarget.hasStdExtM() &&
           "Unexpected custom legalisation");
    // Don't promote division/remainder by constant since we should expand those
    // to multiply by magic constant.
    // FIXME: What if the expansion is disabled for minsize.
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      return;

    // If the input is i32, use ANY_EXTEND since the W instructions don't read
    // the upper 32 bits. For other types we need to sign or zero extend
    // based on the opcode.
    unsigned ExtOpc = ISD::ANY_EXTEND;
    if (VT != MVT::i32)
      ExtOpc = N->getOpcode() == ISD::SDIV ? ISD::SIGN_EXTEND
                                           : ISD::ZERO_EXTEND;

    Results.push_back(customLegalizeToWOp(N, DAG, ExtOpc));
    break;
  }
  case ISD::UADDO:
  case ISD::USUBO: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    bool IsAdd = N->getOpcode() == ISD::UADDO;
    // Create an ADDW or SUBW.
    SDValue LHS = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
    SDValue RHS = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
    SDValue Res =
        DAG.getNode(IsAdd ? ISD::ADD : ISD::SUB, DL, MVT::i64, LHS, RHS);
    Res = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, Res,
                      DAG.getValueType(MVT::i32));

    // Sign extend the LHS and perform an unsigned compare with the ADDW result.
    // Since the inputs are sign extended from i32, this is equivalent to
    // comparing the lower 32 bits.
    LHS = DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, N->getOperand(0));
    SDValue Overflow = DAG.getSetCC(DL, N->getValueType(1), Res, LHS,
                                    IsAdd ? ISD::SETULT : ISD::SETUGT);

    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
    Results.push_back(Overflow);
    return;
  }
  case ISD::UADDSAT:
  case ISD::USUBSAT: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (Subtarget.hasStdExtZbb()) {
      // With Zbb we can sign extend and let LegalizeDAG use minu/maxu. Using
      // sign extend allows overflow of the lower 32 bits to be detected on
      // the promoted size.
      SDValue LHS =
          DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, N->getOperand(0));
      SDValue RHS =
          DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue Res = DAG.getNode(N->getOpcode(), DL, MVT::i64, LHS, RHS);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      return;
    }

    // Without Zbb, expand to UADDO/USUBO+select which will trigger our custom
    // promotion for UADDO/USUBO.
    Results.push_back(expandAddSubSat(N, DAG));
    return;
  }
  case ISD::BITCAST: {
    EVT VT = N->getValueType(0);
    assert(VT.isInteger() && !VT.isVector() && "Unexpected VT!");
    SDValue Op0 = N->getOperand(0);
    EVT Op0VT = Op0.getValueType();
    MVT XLenVT = Subtarget.getXLenVT();
    if (VT == MVT::i16 && Op0VT == MVT::f16 && Subtarget.hasStdExtZfh()) {
      SDValue FPConv = DAG.getNode(PrimateISD::FMV_X_ANYEXTH, DL, XLenVT, Op0);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, FPConv));
    } else if (VT == MVT::i32 && Op0VT == MVT::f32 && Subtarget.is64Bit() &&
               Subtarget.hasStdExtF()) {
      SDValue FPConv =
          DAG.getNode(PrimateISD::FMV_X_ANYEXTW_PR64, DL, MVT::i64, Op0);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, FPConv));
    } else if (!VT.isVector() && Op0VT.isFixedLengthVector() &&
               isTypeLegal(Op0VT)) {
      // Custom-legalize bitcasts from fixed-length vector types to illegal
      // scalar types in order to improve codegen. Bitcast the vector to a
      // one-element vector type whose element type is the same as the result
      // type, and extract the first element.
      LLVMContext &Context = *DAG.getContext();
      SDValue BVec = DAG.getBitcast(EVT::getVectorVT(Context, VT, 1), Op0);
      Results.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, BVec,
                                    DAG.getConstant(0, DL, XLenVT)));
    }
    break;
  }
  case PrimateISD::GREV:
  case PrimateISD::GORC: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    assert(isa<ConstantSDNode>(N->getOperand(1)) && "Expected constant");
    // This is similar to customLegalizeToWOp, except that we pass the second
    // operand (a TargetConstant) straight through: it is already of type
    // XLenVT.
    PrimateISD::NodeType WOpcode = getPrimateWOpcode(N->getOpcode());
    SDValue NewOp0 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
    SDValue NewOp1 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
    SDValue NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp0, NewOp1);
    // ReplaceNodeResults requires we maintain the same type for the return
    // value.
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes));
    break;
  }
  case PrimateISD::SHFL: {
    // There is no SHFLIW instruction, but we can just promote the operation.
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    assert(isa<ConstantSDNode>(N->getOperand(1)) && "Expected constant");
    SDValue NewOp0 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
    SDValue NewOp1 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
    SDValue NewRes = DAG.getNode(PrimateISD::SHFL, DL, MVT::i64, NewOp0, NewOp1);
    // ReplaceNodeResults requires we maintain the same type for the return
    // value.
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes));
    break;
  }
  case ISD::BSWAP:
  case ISD::BITREVERSE: {
    MVT VT = N->getSimpleValueType(0);
    MVT XLenVT = Subtarget.getXLenVT();
    assert((VT == MVT::i8 || VT == MVT::i16 ||
            (VT == MVT::i32 && Subtarget.is64Bit())) &&
           Subtarget.hasStdExtZbb() && "Unexpected custom legalisation");
    SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, XLenVT, N->getOperand(0));
    unsigned Imm = VT.getSizeInBits() - 1;
    // If this is BSWAP rather than BITREVERSE, clear the lower 3 bits.
    if (N->getOpcode() == ISD::BSWAP)
      Imm &= ~0x7U;
    unsigned Opc = Subtarget.is64Bit() ? PrimateISD::GREVW : PrimateISD::GREV;
    SDValue GREVI =
        DAG.getNode(Opc, DL, XLenVT, NewOp0, DAG.getConstant(Imm, DL, XLenVT));
    // ReplaceNodeResults requires we maintain the same type for the return
    // value.
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, GREVI));
    break;
  }
  case ISD::FSHL:
  case ISD::FSHR: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           Subtarget.hasStdExtZbb() && "Unexpected custom legalisation");
    SDValue NewOp0 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
    SDValue NewOp1 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
    SDValue NewOp2 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
    // FSLW/FSRW take a 6 bit shift amount but i32 FSHL/FSHR only use 5 bits.
    // Mask the shift amount to 5 bits.
    NewOp2 = DAG.getNode(ISD::AND, DL, MVT::i64, NewOp2,
                         DAG.getConstant(0x1f, DL, MVT::i64));
    unsigned Opc =
        N->getOpcode() == ISD::FSHL ? PrimateISD::FSLW : PrimateISD::FSRW;
    SDValue NewOp = DAG.getNode(Opc, DL, MVT::i64, NewOp0, NewOp1, NewOp2);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewOp));
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    llvm_unreachable("replace node results EXTRACT_VECTOR_ELT");
    // // Custom-legalize an EXTRACT_VECTOR_ELT where XLEN<SEW, as the SEW element
    // // type is illegal (currently only vXi64 PR32).
    // // With vmv.x.s, when SEW > XLEN, only the least-significant XLEN bits are
    // // transferred to the destination register. We issue two of these from the
    // // upper- and lower- halves of the SEW-bit vector element, slid down to the
    // // first element.
    // SDValue Vec = N->getOperand(0);
    // SDValue Idx = N->getOperand(1);

    // // The vector type hasn't been legalized yet so we can't issue target
    // // specific nodes if it needs legalization.
    // // FIXME: We would manually legalize if it's important.
    // if (!isTypeLegal(Vec.getValueType()))
    //   return;

    // MVT VecVT = Vec.getSimpleValueType();

    // assert(!Subtarget.is64Bit() && N->getValueType(0) == MVT::i64 &&
    //        VecVT.getVectorElementType() == MVT::i64 &&
    //        "Unexpected EXTRACT_VECTOR_ELT legalization");

    // // If this is a fixed vector, we need to convert it to a scalable vector.
    // MVT ContainerVT = VecVT;
    // if (VecVT.isFixedLengthVector()) {
    //   ContainerVT = getContainerForFixedLengthVector(VecVT);
    //   Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
    // }

    // MVT XLenVT = Subtarget.getXLenVT();

    // // Use a VL of 1 to avoid processing more elements than we need.
    // MVT MaskVT = MVT::getVectorVT(MVT::i1, VecVT.getVectorElementCount());
    // SDValue VL = DAG.getConstant(1, DL, XLenVT);
    // SDValue Mask = DAG.getNode(PrimateISD::VMSET_VL, DL, MaskVT, VL);

    // // Unless the index is known to be 0, we must slide the vector down to get
    // // the desired element into index 0.
    // if (!isNullConstant(Idx)) {
    //   Vec = DAG.getNode(PrimateISD::VSLIDEDOWN_VL, DL, ContainerVT,
    //                     DAG.getUNDEF(ContainerVT), Vec, Idx, Mask, VL);
    // }

    // // Extract the lower XLEN bits of the correct vector element.
    // SDValue EltLo = DAG.getNode(PrimateISD::VMV_X_S, DL, XLenVT, Vec);

    // // To extract the upper XLEN bits of the vector element, shift the first
    // // element right by 32 bits and re-extract the lower XLEN bits.
    // SDValue ThirtyTwoV = DAG.getNode(PrimateISD::VMV_V_X_VL, DL, ContainerVT,
    //                                  DAG.getConstant(32, DL, XLenVT), VL);
    // SDValue LShr32 = DAG.getNode(PrimateISD::SRL_VL, DL, ContainerVT, Vec,
    //                              ThirtyTwoV, Mask, VL);

    // SDValue EltHi = DAG.getNode(PrimateISD::VMV_X_S, DL, XLenVT, LShr32);

    // Results.push_back(DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, EltLo, EltHi));
    // break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    dbgs() << "replaceNodeResults for intrinsic wo chain\n";
    unsigned IntNo = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    switch (IntNo) {
    default:
      llvm_unreachable(
          "Don't know how to custom type legalize this intrinsic!");
    case Intrinsic::primate_extract: {
      // check and replace ops
      
      SDValue op1 = N->getOperand(0);
      SDValue op2 = N->getOperand(1);
      SDValue op3 = N->getOperand(2);

      SDValue Res = DAG.getNode(N->getOpcode(), DL, MVT::i32, op1, op2, op3); // simply replace with an i32
      Results.push_back(Res);
      LLVM_DEBUG({
        dbgs() << "Num ops: " << N->getNumOperands() << "\n";
        op1.dump();
        op2.dump();
        op3.dump();
        dbgs() << "Lowered op: ";
        N->dump();
        dbgs() << " to: ";
        Res->dump();
      });
      return;
    }
    case Intrinsic::primate_insert: {
      N->dump();

      SDValue op1 = N->getOperand(0);
      SDValue op2 = N->getOperand(1);
      SDValue op3 = N->getOperand(2);
      
      SDValue Res = DAG.getNode(N->getOpcode(), DL, MVT::Primate_aggregate, op1, op2, op3);

      Res.dump();
      errs() << N->getValueType(0).getEVTString() << "\n";
      errs() << "Primate insert intrinsic result replacement (will fail?)\n";
      Results.push_back(Res);
      return;
    }
    case Intrinsic::primate_orc_b: {
      // Lower to the GORCI encoding for orc.b with the operand extended.
      SDValue NewOp =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
      // If Zbp is enabled, use GORCIW which will sign extend the result.
      unsigned Opc =
          Subtarget.hasStdExtZbb() ? PrimateISD::GORCW : PrimateISD::GORC;
      SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp,
                                DAG.getConstant(7, DL, MVT::i64));
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      return;
    }
    case Intrinsic::primate_grev:
    case Intrinsic::primate_gorc: {
      assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
             "Unexpected custom legalisation");
      SDValue NewOp1 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue NewOp2 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
      unsigned Opc =
          IntNo == Intrinsic::primate_grev ? PrimateISD::GREVW : PrimateISD::GORCW;
      SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp1, NewOp2);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      break;
    }
    case Intrinsic::primate_shfl:
    case Intrinsic::primate_unshfl: {
      assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
             "Unexpected custom legalisation");
      SDValue NewOp1 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue NewOp2 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
      unsigned Opc =
          IntNo == Intrinsic::primate_shfl ? PrimateISD::SHFLW : PrimateISD::UNSHFLW;
      if (isa<ConstantSDNode>(N->getOperand(2))) {
        NewOp2 = DAG.getNode(ISD::AND, DL, MVT::i64, NewOp2,
                             DAG.getConstant(0xf, DL, MVT::i64));
        Opc =
            IntNo == Intrinsic::primate_shfl ? PrimateISD::SHFL : PrimateISD::UNSHFL;
      }
      SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp1, NewOp2);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      break;
    }
    case Intrinsic::primate_bcompress:
    case Intrinsic::primate_bdecompress: {
      assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
             "Unexpected custom legalisation");
      SDValue NewOp1 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue NewOp2 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
      unsigned Opc = IntNo == Intrinsic::primate_bcompress
                         ? PrimateISD::BCOMPRESSW
                         : PrimateISD::BDECOMPRESSW;
      SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp1, NewOp2);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      break;
    }
    case Intrinsic::primate_vmv_x_s: {
      llvm_unreachable("Primate intrincs vmv_x_s");
      // EVT VT = N->getValueType(0);
      // MVT XLenVT = Subtarget.getXLenVT();
      // if (VT.bitsLT(XLenVT)) {
      //   // Simple case just extract using vmv.x.s and truncate.
      //   SDValue Extract = DAG.getNode(PrimateISD::VMV_X_S, DL,
      //                                 Subtarget.getXLenVT(), N->getOperand(1));
      //   Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, Extract));
      //   return;
      // }

      // assert(VT == MVT::i64 && !Subtarget.is64Bit() &&
      //        "Unexpected custom legalization");

      // // We need to do the move in two steps.
      // SDValue Vec = N->getOperand(1);
      // MVT VecVT = Vec.getSimpleValueType();

      // // First extract the lower XLEN bits of the element.
      // SDValue EltLo = DAG.getNode(PrimateISD::VMV_X_S, DL, XLenVT, Vec);

      // // To extract the upper XLEN bits of the vector element, shift the first
      // // element right by 32 bits and re-extract the lower XLEN bits.
      // SDValue VL = DAG.getConstant(1, DL, XLenVT);
      // MVT MaskVT = MVT::getVectorVT(MVT::i1, VecVT.getVectorElementCount());
      // SDValue Mask = DAG.getNode(PrimateISD::VMSET_VL, DL, MaskVT, VL);
      // SDValue ThirtyTwoV = DAG.getNode(PrimateISD::VMV_V_X_VL, DL, VecVT,
      //                                  DAG.getConstant(32, DL, XLenVT), VL);
      // SDValue LShr32 =
      //     DAG.getNode(PrimateISD::SRL_VL, DL, VecVT, Vec, ThirtyTwoV, Mask, VL);
      // SDValue EltHi = DAG.getNode(PrimateISD::VMV_X_S, DL, XLenVT, LShr32);

      // Results.push_back(
      //     DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, EltLo, EltHi));
      // break;
    }
    }
    break;
  }
  case ISD::VECREDUCE_ADD:
  case ISD::VECREDUCE_AND:
  case ISD::VECREDUCE_OR:
  case ISD::VECREDUCE_XOR:
  case ISD::VECREDUCE_SMAX:
  case ISD::VECREDUCE_UMAX:
  case ISD::VECREDUCE_SMIN:
  case ISD::VECREDUCE_UMIN:
    if (SDValue V = lowerVECREDUCE(SDValue(N, 0), DAG))
      Results.push_back(V);
    break;
  // case ISD::FLT_ROUNDS_: {
  //   SDVTList VTs = DAG.getVTList(Subtarget.getXLenVT(), MVT::Other);
  //   SDValue Res = DAG.getNode(ISD::FLT_ROUNDS_, DL, VTs, N->getOperand(0));
  //   Results.push_back(Res.getValue(0));
  //   Results.push_back(Res.getValue(1));
  //   break;
  // }
  }
}

// A structure to hold one of the bit-manipulation patterns below. Together, a
// SHL and non-SHL pattern may form a bit-manipulation pair on a single source:
//   (or (and (shl x, 1), 0xAAAAAAAA),
//       (and (srl x, 1), 0x55555555))
struct PrimateBitmanipPat {
  SDValue Op;
  unsigned ShAmt;
  bool IsSHL;

  bool formsPairWith(const PrimateBitmanipPat &Other) const {
    return Op == Other.Op && ShAmt == Other.ShAmt && IsSHL != Other.IsSHL;
  }
};

// Matches patterns of the form
//   (and (shl x, C2), (C1 << C2))
//   (and (srl x, C2), C1)
//   (shl (and x, C1), C2)
//   (srl (and x, (C1 << C2)), C2)
// Where C2 is a power of 2 and C1 has at least that many leading zeroes.
// The expected masks for each shift amount are specified in BitmanipMasks where
// BitmanipMasks[log2(C2)] specifies the expected C1 value.
// The max allowed shift amount is either XLen/2 or XLen/4 determined by whether
// BitmanipMasks contains 6 or 5 entries assuming that the maximum possible
// XLen is 64.
static std::optional<PrimateBitmanipPat>
matchPrimateBitmanipPat(SDValue Op, ArrayRef<uint64_t> BitmanipMasks) {
  assert((BitmanipMasks.size() == 5 || BitmanipMasks.size() == 6) &&
         "Unexpected number of masks");
  std::optional<uint64_t> Mask;
  // Optionally consume a mask around the shift operation.
  if (Op.getOpcode() == ISD::AND && isa<ConstantSDNode>(Op.getOperand(1))) {
    Mask = Op.getConstantOperandVal(1);
    Op = Op.getOperand(0);
  }
  if (Op.getOpcode() != ISD::SHL && Op.getOpcode() != ISD::SRL)
    return {};
  bool IsSHL = Op.getOpcode() == ISD::SHL;

  if (!isa<ConstantSDNode>(Op.getOperand(1)))
    return {};
  uint64_t ShAmt = Op.getConstantOperandVal(1);

  unsigned Width = Op.getValueType() == MVT::i64 ? 64 : 32;
  if (ShAmt >= Width || !isPowerOf2_64(ShAmt))
    return {};
  // If we don't have enough masks for 64 bit, then we must be trying to
  // match SHFL so we're only allowed to shift 1/4 of the width.
  if (BitmanipMasks.size() == 5 && ShAmt >= (Width / 2))
    return {};

  SDValue Src = Op.getOperand(0);

  // The expected mask is shifted left when the AND is found around SHL
  // patterns.
  //   ((x >> 1) & 0x55555555)
  //   ((x << 1) & 0xAAAAAAAA)
  bool SHLExpMask = IsSHL;

  if (!Mask) {
    // Sometimes LLVM keeps the mask as an operand of the shift, typically when
    // the mask is all ones: consume that now.
    if (Src.getOpcode() == ISD::AND && isa<ConstantSDNode>(Src.getOperand(1))) {
      Mask = Src.getConstantOperandVal(1);
      Src = Src.getOperand(0);
      // The expected mask is now in fact shifted left for SRL, so reverse the
      // decision.
      //   ((x & 0xAAAAAAAA) >> 1)
      //   ((x & 0x55555555) << 1)
      SHLExpMask = !SHLExpMask;
    } else {
      // Use a default shifted mask of all-ones if there's no AND, truncated
      // down to the expected width. This simplifies the logic later on.
      Mask = maskTrailingOnes<uint64_t>(Width);
      *Mask &= (IsSHL ? *Mask << ShAmt : *Mask >> ShAmt);
    }
  }

  unsigned MaskIdx = Log2_32(ShAmt);
  uint64_t ExpMask = BitmanipMasks[MaskIdx] & maskTrailingOnes<uint64_t>(Width);

  if (SHLExpMask)
    ExpMask <<= ShAmt;

  if (Mask != ExpMask)
    return {};

  return PrimateBitmanipPat{Src, (unsigned)ShAmt, IsSHL};
}

// Matches any of the following bit-manipulation patterns:
//   (and (shl x, 1), (0x55555555 << 1))
//   (and (srl x, 1), 0x55555555)
//   (shl (and x, 0x55555555), 1)
//   (srl (and x, (0x55555555 << 1)), 1)
// where the shift amount and mask may vary thus:
//   [1]  = 0x55555555 / 0xAAAAAAAA
//   [2]  = 0x33333333 / 0xCCCCCCCC
//   [4]  = 0x0F0F0F0F / 0xF0F0F0F0
//   [8]  = 0x00FF00FF / 0xFF00FF00
//   [16] = 0x0000FFFF / 0xFFFFFFFF
//   [32] = 0x00000000FFFFFFFF / 0xFFFFFFFF00000000 (for PR64)
static std::optional<PrimateBitmanipPat> matchGREVIPat(SDValue Op) {
  // These are the unshifted masks which we use to match bit-manipulation
  // patterns. They may be shifted left in certain circumstances.
  static const uint64_t BitmanipMasks[] = {
      0x5555555555555555ULL, 0x3333333333333333ULL, 0x0F0F0F0F0F0F0F0FULL,
      0x00FF00FF00FF00FFULL, 0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL};

  return matchPrimateBitmanipPat(Op, BitmanipMasks);
}

// Match the following pattern as a GREVI(W) operation
//   (or (BITMANIP_SHL x), (BITMANIP_SRL x))
static SDValue combineORToGREV(SDValue Op, SelectionDAG &DAG,
                               const PrimateSubtarget &Subtarget) {
  assert(Subtarget.hasStdExtZbb() && "Expected Zbp extenson");
  EVT VT = Op.getValueType();

  if (VT == Subtarget.getXLenVT() || (Subtarget.is64Bit() && VT == MVT::i32)) {
    auto LHS = matchGREVIPat(Op.getOperand(0));
    auto RHS = matchGREVIPat(Op.getOperand(1));
    if (LHS && RHS && LHS->formsPairWith(*RHS)) {
      SDLoc DL(Op);
      return DAG.getNode(PrimateISD::GREV, DL, VT, LHS->Op,
                         DAG.getConstant(LHS->ShAmt, DL, VT));
    }
  }
  return SDValue();
}

// Matches any the following pattern as a GORCI(W) operation
// 1.  (or (GREVI x, shamt), x) if shamt is a power of 2
// 2.  (or x, (GREVI x, shamt)) if shamt is a power of 2
// 3.  (or (or (BITMANIP_SHL x), x), (BITMANIP_SRL x))
// Note that with the variant of 3.,
//     (or (or (BITMANIP_SHL x), (BITMANIP_SRL x)), x)
// the inner pattern will first be matched as GREVI and then the outer
// pattern will be matched to GORC via the first rule above.
// 4.  (or (rotl/rotr x, bitwidth/2), x)
static SDValue combineORToGORC(SDValue Op, SelectionDAG &DAG,
                               const PrimateSubtarget &Subtarget) {
  assert(Subtarget.hasStdExtZbb() && "Expected Zbp extenson");
  EVT VT = Op.getValueType();

  if (VT == Subtarget.getXLenVT() || (Subtarget.is64Bit() && VT == MVT::i32)) {
    SDLoc DL(Op);
    SDValue Op0 = Op.getOperand(0);
    SDValue Op1 = Op.getOperand(1);

    auto MatchOROfReverse = [&](SDValue Reverse, SDValue X) {
      if (Reverse.getOpcode() == PrimateISD::GREV && Reverse.getOperand(0) == X &&
          isa<ConstantSDNode>(Reverse.getOperand(1)) &&
          isPowerOf2_32(Reverse.getConstantOperandVal(1)))
        return DAG.getNode(PrimateISD::GORC, DL, VT, X, Reverse.getOperand(1));
      // We can also form GORCI from ROTL/ROTR by half the bitwidth.
      if ((Reverse.getOpcode() == ISD::ROTL ||
           Reverse.getOpcode() == ISD::ROTR) &&
          Reverse.getOperand(0) == X &&
          isa<ConstantSDNode>(Reverse.getOperand(1))) {
        uint64_t RotAmt = Reverse.getConstantOperandVal(1);
        if (RotAmt == (VT.getSizeInBits() / 2))
          return DAG.getNode(PrimateISD::GORC, DL, VT, X,
                             DAG.getConstant(RotAmt, DL, VT));
      }
      return SDValue();
    };

    // Check for either commutable permutation of (or (GREVI x, shamt), x)
    if (SDValue V = MatchOROfReverse(Op0, Op1))
      return V;
    if (SDValue V = MatchOROfReverse(Op1, Op0))
      return V;

    // OR is commutable so canonicalize its OR operand to the left
    if (Op0.getOpcode() != ISD::OR && Op1.getOpcode() == ISD::OR)
      std::swap(Op0, Op1);
    if (Op0.getOpcode() != ISD::OR)
      return SDValue();
    SDValue OrOp0 = Op0.getOperand(0);
    SDValue OrOp1 = Op0.getOperand(1);
    auto LHS = matchGREVIPat(OrOp0);
    // OR is commutable so swap the operands and try again: x might have been
    // on the left
    if (!LHS) {
      std::swap(OrOp0, OrOp1);
      LHS = matchGREVIPat(OrOp0);
    }
    auto RHS = matchGREVIPat(Op1);
    if (LHS && RHS && LHS->formsPairWith(*RHS) && LHS->Op == OrOp1) {
      return DAG.getNode(PrimateISD::GORC, DL, VT, LHS->Op,
                         DAG.getConstant(LHS->ShAmt, DL, VT));
    }
  }
  return SDValue();
}

// Matches any of the following bit-manipulation patterns:
//   (and (shl x, 1), (0x22222222 << 1))
//   (and (srl x, 1), 0x22222222)
//   (shl (and x, 0x22222222), 1)
//   (srl (and x, (0x22222222 << 1)), 1)
// where the shift amount and mask may vary thus:
//   [1]  = 0x22222222 / 0x44444444
//   [2]  = 0x0C0C0C0C / 0x3C3C3C3C
//   [4]  = 0x00F000F0 / 0x0F000F00
//   [8]  = 0x0000FF00 / 0x00FF0000
//   [16] = 0x00000000FFFF0000 / 0x0000FFFF00000000 (for PR64)
static std::optional<PrimateBitmanipPat> matchSHFLPat(SDValue Op) {
  // These are the unshifted masks which we use to match bit-manipulation
  // patterns. They may be shifted left in certain circumstances.
  static const uint64_t BitmanipMasks[] = {
      0x2222222222222222ULL, 0x0C0C0C0C0C0C0C0CULL, 0x00F000F000F000F0ULL,
      0x0000FF000000FF00ULL, 0x00000000FFFF0000ULL};

  return matchPrimateBitmanipPat(Op, BitmanipMasks);
}

// Match (or (or (SHFL_SHL x), (SHFL_SHR x)), (SHFL_AND x)
static SDValue combineORToSHFL(SDValue Op, SelectionDAG &DAG,
                               const PrimateSubtarget &Subtarget) {
  assert(Subtarget.hasStdExtZbb() && "Expected Zbp extenson");
  EVT VT = Op.getValueType();

  if (VT != MVT::i32 && VT != Subtarget.getXLenVT())
    return SDValue();

  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);

  // Or is commutable so canonicalize the second OR to the LHS.
  if (Op0.getOpcode() != ISD::OR)
    std::swap(Op0, Op1);
  if (Op0.getOpcode() != ISD::OR)
    return SDValue();

  // We found an inner OR, so our operands are the operands of the inner OR
  // and the other operand of the outer OR.
  SDValue A = Op0.getOperand(0);
  SDValue B = Op0.getOperand(1);
  SDValue C = Op1;

  auto Match1 = matchSHFLPat(A);
  auto Match2 = matchSHFLPat(B);

  // If neither matched, we failed.
  if (!Match1 && !Match2)
    return SDValue();

  // We had at least one match. if one failed, try the remaining C operand.
  if (!Match1) {
    std::swap(A, C);
    Match1 = matchSHFLPat(A);
    if (!Match1)
      return SDValue();
  } else if (!Match2) {
    std::swap(B, C);
    Match2 = matchSHFLPat(B);
    if (!Match2)
      return SDValue();
  }
  assert(Match1 && Match2);

  // Make sure our matches pair up.
  if (!Match1->formsPairWith(*Match2))
    return SDValue();

  // All the remains is to make sure C is an AND with the same input, that masks
  // out the bits that are being shuffled.
  if (C.getOpcode() != ISD::AND || !isa<ConstantSDNode>(C.getOperand(1)) ||
      C.getOperand(0) != Match1->Op)
    return SDValue();

  uint64_t Mask = C.getConstantOperandVal(1);

  static const uint64_t BitmanipMasks[] = {
      0x9999999999999999ULL, 0xC3C3C3C3C3C3C3C3ULL, 0xF00FF00FF00FF00FULL,
      0xFF0000FFFF0000FFULL, 0xFFFF00000000FFFFULL,
  };

  unsigned Width = Op.getValueType() == MVT::i64 ? 64 : 32;
  unsigned MaskIdx = Log2_32(Match1->ShAmt);
  uint64_t ExpMask = BitmanipMasks[MaskIdx] & maskTrailingOnes<uint64_t>(Width);

  if (Mask != ExpMask)
    return SDValue();

  SDLoc DL(Op);
  return DAG.getNode(PrimateISD::SHFL, DL, VT, Match1->Op,
                     DAG.getConstant(Match1->ShAmt, DL, VT));
}

// Combine (GREVI (GREVI x, C2), C1) -> (GREVI x, C1^C2) when C1^C2 is
// non-zero, and to x when it is. Any repeated GREVI stage undoes itself.
// Combine (GORCI (GORCI x, C2), C1) -> (GORCI x, C1|C2). Repeated stage does
// not undo itself, but they are redundant.
static SDValue combineGREVI_GORCI(SDNode *N, SelectionDAG &DAG) {
  SDValue Src = N->getOperand(0);

  if (Src.getOpcode() != N->getOpcode())
    return SDValue();

  if (!isa<ConstantSDNode>(N->getOperand(1)) ||
      !isa<ConstantSDNode>(Src.getOperand(1)))
    return SDValue();

  unsigned ShAmt1 = N->getConstantOperandVal(1);
  unsigned ShAmt2 = Src.getConstantOperandVal(1);
  Src = Src.getOperand(0);

  unsigned CombinedShAmt;
  if (N->getOpcode() == PrimateISD::GORC || N->getOpcode() == PrimateISD::GORCW)
    CombinedShAmt = ShAmt1 | ShAmt2;
  else
    CombinedShAmt = ShAmt1 ^ ShAmt2;

  if (CombinedShAmt == 0)
    return Src;

  SDLoc DL(N);
  return DAG.getNode(
      N->getOpcode(), DL, N->getValueType(0), Src,
      DAG.getConstant(CombinedShAmt, DL, N->getOperand(1).getValueType()));
}

// Combine a constant select operand into its use:
//
// (and (select_cc lhs, rhs, cc, -1, c), x)
//   -> (select_cc lhs, rhs, cc, x, (and, x, c))  [AllOnes=1]
// (or  (select_cc lhs, rhs, cc, 0, c), x)
//   -> (select_cc lhs, rhs, cc, x, (or, x, c))  [AllOnes=0]
// (xor (select_cc lhs, rhs, cc, 0, c), x)
//   -> (select_cc lhs, rhs, cc, x, (xor, x, c))  [AllOnes=0]
static SDValue combineSelectCCAndUse(SDNode *N, SDValue Slct, SDValue OtherOp,
                                     SelectionDAG &DAG, bool AllOnes) {
  EVT VT = N->getValueType(0);

  if (Slct.getOpcode() != PrimateISD::SELECT_CC || !Slct.hasOneUse())
    return SDValue();

  auto isZeroOrAllOnes = [](SDValue N, bool AllOnes) {
    return AllOnes ? isAllOnesConstant(N) : isNullConstant(N);
  };

  bool SwapSelectOps;
  SDValue TrueVal = Slct.getOperand(3);
  SDValue FalseVal = Slct.getOperand(4);
  SDValue NonConstantVal;
  if (isZeroOrAllOnes(TrueVal, AllOnes)) {
    SwapSelectOps = false;
    NonConstantVal = FalseVal;
  } else if (isZeroOrAllOnes(FalseVal, AllOnes)) {
    SwapSelectOps = true;
    NonConstantVal = TrueVal;
  } else
    return SDValue();

  // Slct is now know to be the desired identity constant when CC is true.
  TrueVal = OtherOp;
  FalseVal = DAG.getNode(N->getOpcode(), SDLoc(N), VT, OtherOp, NonConstantVal);
  // Unless SwapSelectOps says CC should be false.
  if (SwapSelectOps)
    std::swap(TrueVal, FalseVal);

  return DAG.getNode(PrimateISD::SELECT_CC, SDLoc(N), VT,
                     {Slct.getOperand(0), Slct.getOperand(1),
                      Slct.getOperand(2), TrueVal, FalseVal});
}

// Attempt combineSelectAndUse on each operand of a commutative operator N.
static SDValue combineSelectCCAndUseCommutative(SDNode *N, SelectionDAG &DAG,
                                                bool AllOnes) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  if (SDValue Result = combineSelectCCAndUse(N, N0, N1, DAG, AllOnes))
    return Result;
  if (SDValue Result = combineSelectCCAndUse(N, N1, N0, DAG, AllOnes))
    return Result;
  return SDValue();
}

static SDValue performANDCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const PrimateSubtarget &Subtarget) {
  SelectionDAG &DAG = DCI.DAG;

  // fold (and (select_cc lhs, rhs, cc, -1, y), x) ->
  //      (select lhs, rhs, cc, x, (and x, y))
  return combineSelectCCAndUseCommutative(N, DAG, true);
}

static SDValue performORCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI,
                                const PrimateSubtarget &Subtarget) {
  SelectionDAG &DAG = DCI.DAG;
  if (Subtarget.hasStdExtZbb()) {
    if (auto GREV = combineORToGREV(SDValue(N, 0), DAG, Subtarget))
      return GREV;
    if (auto GORC = combineORToGORC(SDValue(N, 0), DAG, Subtarget))
      return GORC;
    if (auto SHFL = combineORToSHFL(SDValue(N, 0), DAG, Subtarget))
      return SHFL;
  }

  // fold (or (select_cc lhs, rhs, cc, 0, y), x) ->
  //      (select lhs, rhs, cc, x, (or x, y))
  return combineSelectCCAndUseCommutative(N, DAG, false);
}

static SDValue performXORCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const PrimateSubtarget &Subtarget) {
  SelectionDAG &DAG = DCI.DAG;

  // fold (xor (select_cc lhs, rhs, cc, 0, y), x) ->
  //      (select lhs, rhs, cc, x, (xor x, y))
  return combineSelectCCAndUseCommutative(N, DAG, false);
}

// Attempt to turn ANY_EXTEND into SIGN_EXTEND if the input to the ANY_EXTEND
// has users that require SIGN_EXTEND and the SIGN_EXTEND can be done for free
// by an instruction like ADDW/SUBW/MULW. Without this the ANY_EXTEND would be
// removed during type legalization leaving an ADD/SUB/MUL use that won't use
// ADDW/SUBW/MULW.
static SDValue performANY_EXTENDCombine(SDNode *N,
                                        TargetLowering::DAGCombinerInfo &DCI,
                                        const PrimateSubtarget &Subtarget) {
  if (!Subtarget.is64Bit())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;

  SDValue Src = N->getOperand(0);
  EVT VT = N->getValueType(0);
  if (VT != MVT::i64 || Src.getValueType() != MVT::i32)
    return SDValue();

  // The opcode must be one that can implicitly sign_extend.
  // FIXME: Additional opcodes.
  switch (Src.getOpcode()) {
  default:
    return SDValue();
  case ISD::MUL:
    if (!Subtarget.hasStdExtM())
      return SDValue();
    LLVM_FALLTHROUGH;
  case ISD::ADD:
  case ISD::SUB:
    break;
  }

  // Only handle cases where the result is used by a CopyToReg that likely
  // means the value is a liveout of the basic block. This helps prevent
  // infinite combine loops like PR51206.
  if (none_of(N->uses(),
              [](SDNode *User) { return User->getOpcode() == ISD::CopyToReg; }))
    return SDValue();

  SmallVector<SDNode *, 4> SetCCs;
  for (SDNode::use_iterator UI = Src.getNode()->use_begin(),
                            UE = Src.getNode()->use_end();
       UI != UE; ++UI) {
    SDNode *User = *UI;
    if (User == N)
      continue;
    if (UI.getUse().getResNo() != Src.getResNo())
      continue;
    // All i32 setccs are legalized by sign extending operands.
    if (User->getOpcode() == ISD::SETCC) {
      SetCCs.push_back(User);
      continue;
    }
    // We don't know if we can extend this user.
    break;
  }

  // If we don't have any SetCCs, this isn't worthwhile.
  if (SetCCs.empty())
    return SDValue();

  SDLoc DL(N);
  SDValue SExt = DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, Src);
  DCI.CombineTo(N, SExt);

  // Promote all the setccs.
  for (SDNode *SetCC : SetCCs) {
    SmallVector<SDValue, 4> Ops;

    for (unsigned j = 0; j != 2; ++j) {
      SDValue SOp = SetCC->getOperand(j);
      if (SOp == Src)
        Ops.push_back(SExt);
      else
        Ops.push_back(DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, SOp));
    }

    Ops.push_back(SetCC->getOperand(2));
    DCI.CombineTo(SetCC,
                  DAG.getNode(ISD::SETCC, DL, SetCC->getValueType(0), Ops));
  }
  return SDValue(N, 0);
}

SDValue PrimateTargetLowering::PerformDAGCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  switch (N->getOpcode()) {
  default:
    break;
  case PrimateISD::SplitF64: {
    SDValue Op0 = N->getOperand(0);
    // If the input to SplitF64 is just BuildPairF64 then the operation is
    // redundant. Instead, use BuildPairF64's operands directly.
    if (Op0->getOpcode() == PrimateISD::BuildPairF64)
      return DCI.CombineTo(N, Op0.getOperand(0), Op0.getOperand(1));

    SDLoc DL(N);

    // It's cheaper to materialise two 32-bit integers than to load a double
    // from the constant pool and transfer it to integer registers through the
    // stack.
    if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op0)) {
      APInt V = C->getValueAPF().bitcastToAPInt();
      SDValue Lo = DAG.getConstant(V.trunc(32), DL, MVT::i32);
      SDValue Hi = DAG.getConstant(V.lshr(32).trunc(32), DL, MVT::i32);
      return DCI.CombineTo(N, Lo, Hi);
    }

    // This is a target-specific version of a DAGCombine performed in
    // DAGCombiner::visitBITCAST. It performs the equivalent of:
    // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
    // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
    if (!(Op0.getOpcode() == ISD::FNEG || Op0.getOpcode() == ISD::FABS) ||
        !Op0.getNode()->hasOneUse())
      break;
    SDValue NewSplitF64 =
        DAG.getNode(PrimateISD::SplitF64, DL, DAG.getVTList(MVT::i32, MVT::i32),
                    Op0.getOperand(0));
    SDValue Lo = NewSplitF64.getValue(0);
    SDValue Hi = NewSplitF64.getValue(1);
    APInt SignBit = APInt::getSignMask(32);
    if (Op0.getOpcode() == ISD::FNEG) {
      SDValue NewHi = DAG.getNode(ISD::XOR, DL, MVT::i32, Hi,
                                  DAG.getConstant(SignBit, DL, MVT::i32));
      return DCI.CombineTo(N, Lo, NewHi);
    }
    assert(Op0.getOpcode() == ISD::FABS);
    SDValue NewHi = DAG.getNode(ISD::AND, DL, MVT::i32, Hi,
                                DAG.getConstant(~SignBit, DL, MVT::i32));
    return DCI.CombineTo(N, Lo, NewHi);
  }
  case PrimateISD::SLLW:
  case PrimateISD::SRAW:
  case PrimateISD::SRLW:
  case PrimateISD::ROLW:
  case PrimateISD::RORW: {
    // Only the lower 32 bits of LHS and lower 5 bits of RHS are read.
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    APInt LHSMask = APInt::getLowBitsSet(LHS.getValueSizeInBits(), 32);
    APInt RHSMask = APInt::getLowBitsSet(RHS.getValueSizeInBits(), 5);
    if (SimplifyDemandedBits(N->getOperand(0), LHSMask, DCI) ||
        SimplifyDemandedBits(N->getOperand(1), RHSMask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }
    break;
  }
  case PrimateISD::CLZW:
  case PrimateISD::CTZW: {
    // Only the lower 32 bits of the first operand are read
    SDValue Op0 = N->getOperand(0);
    APInt Mask = APInt::getLowBitsSet(Op0.getValueSizeInBits(), 32);
    if (SimplifyDemandedBits(Op0, Mask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }
    break;
  }
  case PrimateISD::FSL:
  case PrimateISD::FSR: {
    // Only the lower log2(Bitwidth)+1 bits of the the shift amount are read.
    SDValue ShAmt = N->getOperand(2);
    unsigned BitWidth = ShAmt.getValueSizeInBits();
    assert(isPowerOf2_32(BitWidth) && "Unexpected bit width");
    APInt ShAmtMask(BitWidth, (BitWidth * 2) - 1);
    if (SimplifyDemandedBits(ShAmt, ShAmtMask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }
    break;
  }
  case PrimateISD::FSLW:
  case PrimateISD::FSRW: {
    // Only the lower 32 bits of Values and lower 6 bits of shift amount are
    // read.
    SDValue Op0 = N->getOperand(0);
    SDValue Op1 = N->getOperand(1);
    SDValue ShAmt = N->getOperand(2);
    APInt OpMask = APInt::getLowBitsSet(Op0.getValueSizeInBits(), 32);
    APInt ShAmtMask = APInt::getLowBitsSet(ShAmt.getValueSizeInBits(), 6);
    if (SimplifyDemandedBits(Op0, OpMask, DCI) ||
        SimplifyDemandedBits(Op1, OpMask, DCI) ||
        SimplifyDemandedBits(ShAmt, ShAmtMask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }
    break;
  }
  case PrimateISD::GREV:
  case PrimateISD::GORC: {
    // Only the lower log2(Bitwidth) bits of the the shift amount are read.
    SDValue ShAmt = N->getOperand(1);
    unsigned BitWidth = ShAmt.getValueSizeInBits();
    assert(isPowerOf2_32(BitWidth) && "Unexpected bit width");
    APInt ShAmtMask(BitWidth, BitWidth - 1);
    if (SimplifyDemandedBits(ShAmt, ShAmtMask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }

    return combineGREVI_GORCI(N, DCI.DAG);
  }
  case PrimateISD::GREVW:
  case PrimateISD::GORCW: {
    // Only the lower 32 bits of LHS and lower 5 bits of RHS are read.
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    APInt LHSMask = APInt::getLowBitsSet(LHS.getValueSizeInBits(), 32);
    APInt RHSMask = APInt::getLowBitsSet(RHS.getValueSizeInBits(), 5);
    if (SimplifyDemandedBits(LHS, LHSMask, DCI) ||
        SimplifyDemandedBits(RHS, RHSMask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }

    return combineGREVI_GORCI(N, DCI.DAG);
  }
  case PrimateISD::SHFL:
  case PrimateISD::UNSHFL: {
    // Only the lower log2(Bitwidth) bits of the the shift amount are read.
    SDValue ShAmt = N->getOperand(1);
    unsigned BitWidth = ShAmt.getValueSizeInBits();
    assert(isPowerOf2_32(BitWidth) && "Unexpected bit width");
    APInt ShAmtMask(BitWidth, (BitWidth / 2) - 1);
    if (SimplifyDemandedBits(ShAmt, ShAmtMask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }

    break;
  }
  case PrimateISD::SHFLW:
  case PrimateISD::UNSHFLW: {
    // Only the lower 32 bits of LHS and lower 5 bits of RHS are read.
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    APInt LHSMask = APInt::getLowBitsSet(LHS.getValueSizeInBits(), 32);
    APInt RHSMask = APInt::getLowBitsSet(RHS.getValueSizeInBits(), 4);
    if (SimplifyDemandedBits(LHS, LHSMask, DCI) ||
        SimplifyDemandedBits(RHS, RHSMask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }

    break;
  }
  case PrimateISD::BCOMPRESSW:
  case PrimateISD::BDECOMPRESSW: {
    // Only the lower 32 bits of LHS and RHS are read.
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    APInt Mask = APInt::getLowBitsSet(LHS.getValueSizeInBits(), 32);
    if (SimplifyDemandedBits(LHS, Mask, DCI) ||
        SimplifyDemandedBits(RHS, Mask, DCI)) {
      if (N->getOpcode() != ISD::DELETED_NODE)
        DCI.AddToWorklist(N);
      return SDValue(N, 0);
    }

    break;
  }
  case PrimateISD::FMV_X_ANYEXTW_PR64: {
    SDLoc DL(N);
    SDValue Op0 = N->getOperand(0);
    // If the input to FMV_X_ANYEXTW_PR64 is just FMV_W_X_PR64 then the
    // conversion is unnecessary and can be replaced with an ANY_EXTEND
    // of the FMV_W_X_PR64 operand.
    if (Op0->getOpcode() == PrimateISD::FMV_W_X_PR64) {
      assert(Op0.getOperand(0).getValueType() == MVT::i64 &&
             "Unexpected value type!");
      return Op0.getOperand(0);
    }

    // This is a target-specific version of a DAGCombine performed in
    // DAGCombiner::visitBITCAST. It performs the equivalent of:
    // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
    // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
    if (!(Op0.getOpcode() == ISD::FNEG || Op0.getOpcode() == ISD::FABS) ||
        !Op0.getNode()->hasOneUse())
      break;
    SDValue NewFMV = DAG.getNode(PrimateISD::FMV_X_ANYEXTW_PR64, DL, MVT::i64,
                                 Op0.getOperand(0));
    APInt SignBit = APInt::getSignMask(32).sext(64);
    if (Op0.getOpcode() == ISD::FNEG)
      return DAG.getNode(ISD::XOR, DL, MVT::i64, NewFMV,
                         DAG.getConstant(SignBit, DL, MVT::i64));

    assert(Op0.getOpcode() == ISD::FABS);
    return DAG.getNode(ISD::AND, DL, MVT::i64, NewFMV,
                       DAG.getConstant(~SignBit, DL, MVT::i64));
  }
  case ISD::AND:
    return performANDCombine(N, DCI, Subtarget);
  case ISD::OR:
    return performORCombine(N, DCI, Subtarget);
  case ISD::XOR:
    return performXORCombine(N, DCI, Subtarget);
  case ISD::ANY_EXTEND:
    return performANY_EXTENDCombine(N, DCI, Subtarget);
  case ISD::ZERO_EXTEND:
    // Fold (zero_extend (fp_to_uint X)) to prevent forming fcvt+zexti32 during
    // type legalization. This is safe because fp_to_uint produces poison if
    // it overflows.
    if (N->getValueType(0) == MVT::i64 && Subtarget.is64Bit() &&
        N->getOperand(0).getOpcode() == ISD::FP_TO_UINT &&
        isTypeLegal(N->getOperand(0).getOperand(0).getValueType()))
      return DAG.getNode(ISD::FP_TO_UINT, SDLoc(N), MVT::i64,
                         N->getOperand(0).getOperand(0));
    return SDValue();
  case PrimateISD::SELECT_CC: {
    // Transform
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    auto CCVal = static_cast<ISD::CondCode>(N->getConstantOperandVal(2));
    if (!ISD::isIntEqualitySetCC(CCVal))
      break;

    // Fold (select_cc (setlt X, Y), 0, ne, trueV, falseV) ->
    //      (select_cc X, Y, lt, trueV, falseV)
    // Sometimes the setcc is introduced after select_cc has been formed.
    if (LHS.getOpcode() == ISD::SETCC && isNullConstant(RHS) &&
        LHS.getOperand(0).getValueType() == Subtarget.getXLenVT()) {
      // If we're looking for eq 0 instead of ne 0, we need to invert the
      // condition.
      bool Invert = CCVal == ISD::SETEQ;
      CCVal = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
      if (Invert)
        CCVal = ISD::getSetCCInverse(CCVal, LHS.getValueType());

      SDLoc DL(N);
      RHS = LHS.getOperand(1);
      LHS = LHS.getOperand(0);
      translateSetCCForBranch(DL, LHS, RHS, CCVal, DAG);

      SDValue TargetCC =
          DAG.getTargetConstant(CCVal, DL, Subtarget.getXLenVT());
      return DAG.getNode(
          PrimateISD::SELECT_CC, DL, N->getValueType(0),
          {LHS, RHS, TargetCC, N->getOperand(3), N->getOperand(4)});
    }

    // Fold (select_cc (xor X, Y), 0, eq/ne, trueV, falseV) ->
    //      (select_cc X, Y, eq/ne, trueV, falseV)
    if (LHS.getOpcode() == ISD::XOR && isNullConstant(RHS))
      return DAG.getNode(PrimateISD::SELECT_CC, SDLoc(N), N->getValueType(0),
                         {LHS.getOperand(0), LHS.getOperand(1),
                          N->getOperand(2), N->getOperand(3),
                          N->getOperand(4)});
    // (select_cc X, 1, setne, trueV, falseV) ->
    // (select_cc X, 0, seteq, trueV, falseV) if we can prove X is 0/1.
    // This can occur when legalizing some floating point comparisons.
    APInt Mask = APInt::getBitsSetFrom(LHS.getValueSizeInBits(), 1);
    if (isOneConstant(RHS) && DAG.MaskedValueIsZero(LHS, Mask)) {
      SDLoc DL(N);
      CCVal = ISD::getSetCCInverse(CCVal, LHS.getValueType());
      SDValue TargetCC =
          DAG.getTargetConstant(CCVal, DL, Subtarget.getXLenVT());
      RHS = DAG.getConstant(0, DL, LHS.getValueType());
      return DAG.getNode(
          PrimateISD::SELECT_CC, DL, N->getValueType(0),
          {LHS, RHS, TargetCC, N->getOperand(3), N->getOperand(4)});
    }

    break;
  }
  case PrimateISD::BR_CC: {
    SDValue LHS = N->getOperand(1);
    SDValue RHS = N->getOperand(2);
    ISD::CondCode CCVal = cast<CondCodeSDNode>(N->getOperand(3))->get();
    if (!ISD::isIntEqualitySetCC(CCVal))
      break;

    // Fold (br_cc (setlt X, Y), 0, ne, dest) ->
    //      (br_cc X, Y, lt, dest)
    // Sometimes the setcc is introduced after br_cc has been formed.
    if (LHS.getOpcode() == ISD::SETCC && isNullConstant(RHS) &&
        LHS.getOperand(0).getValueType() == Subtarget.getXLenVT()) {
      // If we're looking for eq 0 instead of ne 0, we need to invert the
      // condition.
      bool Invert = CCVal == ISD::SETEQ;
      CCVal = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
      if (Invert)
        CCVal = ISD::getSetCCInverse(CCVal, LHS.getValueType());

      SDLoc DL(N);
      RHS = LHS.getOperand(1);
      LHS = LHS.getOperand(0);
      translateSetCCForBranch(DL, LHS, RHS, CCVal, DAG);

      return DAG.getNode(PrimateISD::BR_CC, DL, N->getValueType(0),
                         N->getOperand(0), LHS, RHS, DAG.getCondCode(CCVal),
                         N->getOperand(4));
    }

    // Fold (br_cc (xor X, Y), 0, eq/ne, dest) ->
    //      (br_cc X, Y, eq/ne, trueV, falseV)
    if (LHS.getOpcode() == ISD::XOR && isNullConstant(RHS))
      return DAG.getNode(PrimateISD::BR_CC, SDLoc(N), N->getValueType(0),
                         N->getOperand(0), LHS.getOperand(0), LHS.getOperand(1),
                         N->getOperand(3), N->getOperand(4));

    // (br_cc X, 1, setne, br_cc) ->
    // (br_cc X, 0, seteq, br_cc) if we can prove X is 0/1.
    // This can occur when legalizing some floating point comparisons.
    APInt Mask = APInt::getBitsSetFrom(LHS.getValueSizeInBits(), 1);
    if (isOneConstant(RHS) && DAG.MaskedValueIsZero(LHS, Mask)) {
      SDLoc DL(N);
      CCVal = ISD::getSetCCInverse(CCVal, LHS.getValueType());
      SDValue TargetCC = DAG.getCondCode(CCVal);
      RHS = DAG.getConstant(0, DL, LHS.getValueType());
      return DAG.getNode(PrimateISD::BR_CC, DL, N->getValueType(0),
                         N->getOperand(0), LHS, RHS, TargetCC,
                         N->getOperand(4));
    }
    break;
  }
  case ISD::FCOPYSIGN: {
    EVT VT = N->getValueType(0);
    if (!VT.isVector())
      break;
    // There is a form of VFSGNJ which injects the negated sign of its second
    // operand. Try and bubble any FNEG up after the extend/round to produce
    // this optimized pattern. Avoid modifying cases where FP_ROUND and
    // TRUNC=1.
    SDValue In2 = N->getOperand(1);
    // Avoid cases where the extend/round has multiple uses, as duplicating
    // those is typically more expensive than removing a fneg.
    if (!In2.hasOneUse())
      break;
    if (In2.getOpcode() != ISD::FP_EXTEND &&
        (In2.getOpcode() != ISD::FP_ROUND || In2.getConstantOperandVal(1) != 0))
      break;
    In2 = In2.getOperand(0);
    if (In2.getOpcode() != ISD::FNEG)
      break;
    SDLoc DL(N);
    SDValue NewFPExtRound = DAG.getFPExtendOrRound(In2.getOperand(0), DL, VT);
    return DAG.getNode(ISD::FCOPYSIGN, DL, VT, N->getOperand(0),
                       DAG.getNode(ISD::FNEG, DL, VT, NewFPExtRound));
  }
  case ISD::MGATHER:
  case ISD::MSCATTER: {
    if (!DCI.isBeforeLegalize())
      break;
    MaskedGatherScatterSDNode *MGSN = cast<MaskedGatherScatterSDNode>(N);
    SDValue Index = MGSN->getIndex();
    EVT IndexVT = Index.getValueType();
    MVT XLenVT = Subtarget.getXLenVT();
    // Primate indexed loads only support the "unsigned unscaled" addressing
    // mode, so anything else must be manually legalized.
    bool NeedsIdxLegalization = MGSN->isIndexScaled() ||
                                (MGSN->isIndexSigned() &&
                                 IndexVT.getVectorElementType().bitsLT(XLenVT));
    if (!NeedsIdxLegalization)
      break;

    SDLoc DL(N);

    // Any index legalization should first promote to XLenVT, so we don't lose
    // bits when scaling. This may create an illegal index type so we let
    // LLVM's legalization take care of the splitting.
    if (IndexVT.getVectorElementType().bitsLT(XLenVT)) {
      IndexVT = IndexVT.changeVectorElementType(XLenVT);
      Index = DAG.getNode(MGSN->isIndexSigned() ? ISD::SIGN_EXTEND
                                                : ISD::ZERO_EXTEND,
                          DL, IndexVT, Index);
    }

    unsigned Scale = N->getConstantOperandVal(5);
    if (MGSN->isIndexScaled() && Scale != 1) {
      // Manually scale the indices by the element size.
      // TODO: Sanitize the scale operand here?
      assert(isPowerOf2_32(Scale) && "Expecting power-of-two types");
      SDValue SplatScale = DAG.getConstant(Log2_32(Scale), DL, IndexVT);
      Index = DAG.getNode(ISD::SHL, DL, IndexVT, Index, SplatScale);
    }

    ISD::MemIndexType NewIndexTy = ISD::MemIndexType::UNSIGNED_SCALED;
    if (const auto *MGN = dyn_cast<MaskedGatherSDNode>(N)) {
      return DAG.getMaskedGather(
          N->getVTList(), MGSN->getMemoryVT(), DL,
          {MGSN->getChain(), MGN->getPassThru(), MGSN->getMask(),
           MGSN->getBasePtr(), Index, MGN->getScale()},
          MGN->getMemOperand(), NewIndexTy, MGN->getExtensionType());
    }
    const auto *MSN = cast<MaskedScatterSDNode>(N);
    return DAG.getMaskedScatter(
        N->getVTList(), MGSN->getMemoryVT(), DL,
        {MGSN->getChain(), MSN->getValue(), MGSN->getMask(), MGSN->getBasePtr(),
         Index, MGSN->getScale()},
        MGSN->getMemOperand(), NewIndexTy, MSN->isTruncatingStore());
  }
  // case PrimateISD::SRA_VL:
  // case PrimateISD::SRL_VL:
  // case PrimateISD::SHL_VL: {
  //   SDValue ShAmt = N->getOperand(1);
  //   if (ShAmt.getOpcode() == PrimateISD::SPLAT_VECTOR_SPLIT_I64_VL) {
  //     // We don't need the upper 32 bits of a 64-bit element for a shift amount.
  //     SDLoc DL(N);
  //     SDValue VL = N->getOperand(3);
  //     EVT VT = N->getValueType(0);
  //     ShAmt =
  //         DAG.getNode(PrimateISD::VMV_V_X_VL, DL, VT, ShAmt.getOperand(0), VL);
  //     return DAG.getNode(N->getOpcode(), DL, VT, N->getOperand(0), ShAmt,
  //                        N->getOperand(2), N->getOperand(3));
  //   }
  //   break;
  // }
  // case ISD::SRA:
  // case ISD::SRL:
  // case ISD::SHL: {
  //   SDValue ShAmt = N->getOperand(1);
  //   if (ShAmt.getOpcode() == PrimateISD::SPLAT_VECTOR_SPLIT_I64_VL) {
  //     // We don't need the upper 32 bits of a 64-bit element for a shift amount.
  //     SDLoc DL(N);
  //     EVT VT = N->getValueType(0);
  //     ShAmt =
  //         DAG.getNode(PrimateISD::SPLAT_VECTOR_I64, DL, VT, ShAmt.getOperand(0));
  //     return DAG.getNode(N->getOpcode(), DL, VT, N->getOperand(0), ShAmt);
  //   }
  //   break;
  // }
  // case PrimateISD::MUL_VL: {
  //   // Try to form VWMUL or VWMULU.
  //   // FIXME: Look for splat of extended scalar as well.
  //   // FIXME: Support VWMULSU.
  //   SDValue Op0 = N->getOperand(0);
  //   SDValue Op1 = N->getOperand(1);
  //   bool IsSignExt = Op0.getOpcode() == PrimateISD::VSEXT_VL;
  //   bool IsZeroExt = Op0.getOpcode() == PrimateISD::VZEXT_VL;
  //   if ((!IsSignExt && !IsZeroExt) || Op0.getOpcode() != Op1.getOpcode())
  //     return SDValue();

  //   // Make sure the extends have a single use.
  //   if (!Op0.hasOneUse() || !Op1.hasOneUse())
  //     return SDValue();

  //   SDValue Mask = N->getOperand(2);
  //   SDValue VL = N->getOperand(3);
  //   if (Op0.getOperand(1) != Mask || Op1.getOperand(1) != Mask ||
  //       Op0.getOperand(2) != VL || Op1.getOperand(2) != VL)
  //     return SDValue();

  //   Op0 = Op0.getOperand(0);
  //   Op1 = Op1.getOperand(0);

  //   MVT VT = N->getSimpleValueType(0);
  //   MVT NarrowVT =
  //       MVT::getVectorVT(MVT::getIntegerVT(VT.getScalarSizeInBits() / 2),
  //                        VT.getVectorElementCount());

  //   SDLoc DL(N);

  //   // Re-introduce narrower extends if needed.
  //   unsigned ExtOpc = IsSignExt ? PrimateISD::VSEXT_VL : PrimateISD::VZEXT_VL;
  //   if (Op0.getValueType() != NarrowVT)
  //     Op0 = DAG.getNode(ExtOpc, DL, NarrowVT, Op0, Mask, VL);
  //   if (Op1.getValueType() != NarrowVT)
  //     Op1 = DAG.getNode(ExtOpc, DL, NarrowVT, Op1, Mask, VL);

  //   unsigned WMulOpc = IsSignExt ? PrimateISD::VWMUL_VL : PrimateISD::VWMULU_VL;
  //   return DAG.getNode(WMulOpc, DL, VT, Op0, Op1, Mask, VL);
  // }
  }

  return SDValue();
}

bool PrimateTargetLowering::isDesirableToCommuteWithShift(
    const SDNode *N, CombineLevel Level) const {
  // The following folds are only desirable if `(OP _, c1 << c2)` can be
  // materialised in fewer instructions than `(OP _, c1)`:
  //
  //   (shl (add x, c1), c2) -> (add (shl x, c2), c1 << c2)
  //   (shl (or x, c1), c2) -> (or (shl x, c2), c1 << c2)
  SDValue N0 = N->getOperand(0);
  EVT Ty = N0.getValueType();
  if (Ty.isScalarInteger() &&
      (N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::OR)) {
    auto *C1 = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    auto *C2 = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (C1 && C2) {
      const APInt &C1Int = C1->getAPIntValue();
      APInt ShiftedC1Int = C1Int << C2->getAPIntValue();

      // We can materialise `c1 << c2` into an add immediate, so it's "free",
      // and the combine should happen, to potentially allow further combines
      // later.
      if (ShiftedC1Int.getSignificantBits() <= 64 &&
          isLegalAddImmediate(ShiftedC1Int.getSExtValue()))
        return true;

      // We can materialise `c1` in an add immediate, so it's "free", and the
      // combine should be prevented.
      if (C1Int.getSignificantBits() <= 64 &&
          isLegalAddImmediate(C1Int.getSExtValue()))
        return false;

      // Neither constant will fit into an immediate, so find materialisation
      // costs.
      int C1Cost = PrimateMatInt::getIntMatCost(C1Int, Ty.getSizeInBits(),
                                              Subtarget,
                                              /*CompressionCost*/true);
      int ShiftedC1Cost = PrimateMatInt::getIntMatCost(
          ShiftedC1Int, Ty.getSizeInBits(), Subtarget,
          /*CompressionCost*/true);

      // Materialising `c1` is cheaper than materialising `c1 << c2`, so the
      // combine should be prevented.
      if (C1Cost < ShiftedC1Cost)
        return false;
    }
  }
  return true;
}

bool PrimateTargetLowering::targetShrinkDemandedConstant(
    SDValue Op, const APInt &DemandedBits, const APInt &DemandedElts,
    TargetLoweringOpt &TLO) const {
  // Delay this optimization as late as possible.
  if (!TLO.LegalOps)
    return false;

  EVT VT = Op.getValueType();
  if (VT.isVector())
    return false;

  // Only handle AND for now.
  if (Op.getOpcode() != ISD::AND)
    return false;

  ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
  if (!C)
    return false;

  const APInt &Mask = C->getAPIntValue();

  // Clear all non-demanded bits initially.
  APInt ShrunkMask = Mask & DemandedBits;

  // Try to make a smaller immediate by setting undemanded bits.

  APInt ExpandedMask = Mask | ~DemandedBits;

  auto IsLegalMask = [ShrunkMask, ExpandedMask](const APInt &Mask) -> bool {
    return ShrunkMask.isSubsetOf(Mask) && Mask.isSubsetOf(ExpandedMask);
  };
  auto UseMask = [Mask, Op, VT, &TLO](const APInt &NewMask) -> bool {
    if (NewMask == Mask)
      return true;
    SDLoc DL(Op);
    SDValue NewC = TLO.DAG.getConstant(NewMask, DL, VT);
    SDValue NewOp = TLO.DAG.getNode(ISD::AND, DL, VT, Op.getOperand(0), NewC);
    return TLO.CombineTo(Op, NewOp);
  };

  // If the shrunk mask fits in sign extended 12 bits, let the target
  // independent code apply it.
  if (ShrunkMask.isSignedIntN(12))
    return false;

  // Preserve (and X, 0xffff) when zext.h is supported.
  if (Subtarget.hasStdExtZbb() || Subtarget.hasStdExtZbb()) {
    APInt NewMask = APInt(Mask.getBitWidth(), 0xffff);
    if (IsLegalMask(NewMask))
      return UseMask(NewMask);
  }

  // Try to preserve (and X, 0xffffffff), the (zext_inreg X, i32) pattern.
  if (VT == MVT::i64) {
    APInt NewMask = APInt(64, 0xffffffff);
    if (IsLegalMask(NewMask))
      return UseMask(NewMask);
  }

  // For the remaining optimizations, we need to be able to make a negative
  // number through a combination of mask and undemanded bits.
  if (!ExpandedMask.isNegative())
    return false;

  // What is the fewest number of bits we need to represent the negative number.
  unsigned MinSignedBits = ExpandedMask.getSignificantBits();

  // Try to make a 12 bit negative immediate. If that fails try to make a 32
  // bit negative immediate unless the shrunk immediate already fits in 32 bits.
  APInt NewMask = ShrunkMask;
  if (MinSignedBits <= 12)
    NewMask.setBitsFrom(11);
  else if (MinSignedBits <= 32 && !ShrunkMask.isSignedIntN(32))
    NewMask.setBitsFrom(31);
  else
    return false;

  // Sanity check that our new mask is a subset of the demanded mask.
  assert(IsLegalMask(NewMask));
  return UseMask(NewMask);
}

static void computeGREV(APInt &Src, unsigned ShAmt) {
  ShAmt &= Src.getBitWidth() - 1;
  uint64_t x = Src.getZExtValue();
  if (ShAmt & 1)
    x = ((x & 0x5555555555555555LL) << 1) | ((x & 0xAAAAAAAAAAAAAAAALL) >> 1);
  if (ShAmt & 2)
    x = ((x & 0x3333333333333333LL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCLL) >> 2);
  if (ShAmt & 4)
    x = ((x & 0x0F0F0F0F0F0F0F0FLL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0LL) >> 4);
  if (ShAmt & 8)
    x = ((x & 0x00FF00FF00FF00FFLL) << 8) | ((x & 0xFF00FF00FF00FF00LL) >> 8);
  if (ShAmt & 16)
    x = ((x & 0x0000FFFF0000FFFFLL) << 16) | ((x & 0xFFFF0000FFFF0000LL) >> 16);
  if (ShAmt & 32)
    x = ((x & 0x00000000FFFFFFFFLL) << 32) | ((x & 0xFFFFFFFF00000000LL) >> 32);
  Src = x;
}

void PrimateTargetLowering::computeKnownBitsForTargetNode(const SDValue Op,
                                                        KnownBits &Known,
                                                        const APInt &DemandedElts,
                                                        const SelectionDAG &DAG,
                                                        unsigned Depth) const {
  unsigned BitWidth = Known.getBitWidth();
  unsigned Opc = Op.getOpcode();
  assert((Opc >= ISD::BUILTIN_OP_END ||
          Opc == ISD::INTRINSIC_WO_CHAIN ||
          Opc == ISD::INTRINSIC_W_CHAIN ||
          Opc == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");

  Known.resetAll();
  switch (Opc) {
  default: break;
  case PrimateISD::SELECT_CC: {
    Known = DAG.computeKnownBits(Op.getOperand(4), Depth + 1);
    // If we don't know any bits, early out.
    if (Known.isUnknown())
      break;
    KnownBits Known2 = DAG.computeKnownBits(Op.getOperand(3), Depth + 1);

    // Only known if known in both the LHS and RHS.
    Known = Known.intersectWith(Known2);
    break;
  }
  case PrimateISD::REMUW: {
    KnownBits Known2;
    Known = DAG.computeKnownBits(Op.getOperand(0), DemandedElts, Depth + 1);
    Known2 = DAG.computeKnownBits(Op.getOperand(1), DemandedElts, Depth + 1);
    // We only care about the lower 32 bits.
    Known = KnownBits::urem(Known.trunc(32), Known2.trunc(32));
    // Restore the original width by sign extending.
    Known = Known.sext(BitWidth);
    break;
  }
  case PrimateISD::DIVUW: {
    KnownBits Known2;
    Known = DAG.computeKnownBits(Op.getOperand(0), DemandedElts, Depth + 1);
    Known2 = DAG.computeKnownBits(Op.getOperand(1), DemandedElts, Depth + 1);
    // We only care about the lower 32 bits.
    Known = KnownBits::udiv(Known.trunc(32), Known2.trunc(32));
    // Restore the original width by sign extending.
    Known = Known.sext(BitWidth);
    break;
  }
  case PrimateISD::CTZW: {
    KnownBits Known2 = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    unsigned PossibleTZ = Known2.trunc(32).countMaxTrailingZeros();
    unsigned LowBits = Log2_32(PossibleTZ) + 1;
    Known.Zero.setBitsFrom(LowBits);
    break;
  }
  case PrimateISD::CLZW: {
    KnownBits Known2 = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    unsigned PossibleLZ = Known2.trunc(32).countMaxLeadingZeros();
    unsigned LowBits = Log2_32(PossibleLZ) + 1;
    Known.Zero.setBitsFrom(LowBits);
    break;
  }
  case PrimateISD::GREV:
  case PrimateISD::GREVW: {
    if (auto *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      Known = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
      if (Opc == PrimateISD::GREVW)
        Known = Known.trunc(32);
      unsigned ShAmt = C->getZExtValue();
      computeGREV(Known.Zero, ShAmt);
      computeGREV(Known.One, ShAmt);
      if (Opc == PrimateISD::GREVW)
        Known = Known.sext(BitWidth);
    }
    break;
  }
  // case PrimateISD::READ_VLENB:
  //   // We assume VLENB is at least 16 bytes.
  //   Known.Zero.setLowBits(4);
  //   // We assume VLENB is no more than 65536 / 8 bytes.
  //   Known.Zero.setBitsFrom(14);
  //   break;
  case ISD::INTRINSIC_W_CHAIN: {
    LLVM_DEBUG(dbgs () << "Int w chain 1\n");
    unsigned IntNo = Op.getConstantOperandVal(1);
    switch (IntNo) {
    default:
      // We can't do anything for most intrinsics.
      break;
    }
    break;
  }
  }
}

unsigned PrimateTargetLowering::ComputeNumSignBitsForTargetNode(
    SDValue Op, const APInt &DemandedElts, const SelectionDAG &DAG,
    unsigned Depth) const {
  switch (Op.getOpcode()) {
  default:
    break;
  case PrimateISD::SLLW:
  case PrimateISD::SRAW:
  case PrimateISD::SRLW:
  case PrimateISD::DIVW:
  case PrimateISD::DIVUW:
  case PrimateISD::REMUW:
  case PrimateISD::ROLW:
  case PrimateISD::RORW:
  case PrimateISD::GREVW:
  case PrimateISD::GORCW:
  case PrimateISD::FSLW:
  case PrimateISD::FSRW:
  case PrimateISD::SHFLW:
  case PrimateISD::UNSHFLW:
  case PrimateISD::BCOMPRESSW:
  case PrimateISD::BDECOMPRESSW:
  case PrimateISD::FCVT_W_PR64:
  case PrimateISD::FCVT_WU_PR64:
    // TODO: As the result is sign-extended, this is conservatively correct. A
    // more precise answer could be calculated for SRAW depending on known
    // bits in the shift amount.
    return 33;
  case PrimateISD::SHFL:
  case PrimateISD::UNSHFL: {
    // There is no SHFLIW, but a i64 SHFLI with bit 4 of the control word
    // cleared doesn't affect bit 31. The upper 32 bits will be shuffled, but
    // will stay within the upper 32 bits. If there were more than 32 sign bits
    // before there will be at least 33 sign bits after.
    if (Op.getValueType() == MVT::i64 &&
        isa<ConstantSDNode>(Op.getOperand(1)) &&
        (Op.getConstantOperandVal(1) & 0x10) == 0) {
      unsigned Tmp = DAG.ComputeNumSignBits(Op.getOperand(0), Depth + 1);
      if (Tmp > 32)
        return 33;
    }
    break;
  }
  // case PrimateISD::VMV_X_S:
  //   // The number of sign bits of the scalar result is computed by obtaining the
  //   // element type of the input vector operand, subtracting its width from the
  //   // XLEN, and then adding one (sign bit within the element type). If the
  //   // element type is wider than XLen, the least-significant XLEN bits are
  //   // taken.
  //   if (Op.getOperand(0).getScalarValueSizeInBits() > Subtarget.getXLen())
  //     return 1;
  //   return Subtarget.getXLen() - Op.getOperand(0).getScalarValueSizeInBits() + 1;
  }

  return 1;
}

static MachineBasicBlock *emitReadCycleWidePseudo(MachineInstr &MI,
                                                  MachineBasicBlock *BB) {
  assert(MI.getOpcode() == Primate::ReadCycleWide && "Unexpected instruction");

  // To read the 64-bit cycle CSR on a 32-bit target, we read the two halves.
  // Should the count have wrapped while it was being read, we need to try
  // again.
  // ...
  // read:
  // rdcycleh x3 # load high word of cycle
  // rdcycle  x2 # load low word of cycle
  // rdcycleh x4 # load high word of cycle
  // bne x3, x4, read # check if high word reads match, otherwise try again
  // ...

  MachineFunction &MF = *BB->getParent();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *LoopMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MF.insert(It, LoopMBB);

  MachineBasicBlock *DoneMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MF.insert(It, DoneMBB);

  // Transfer the remainder of BB and its successor edges to DoneMBB.
  DoneMBB->splice(DoneMBB->begin(), BB,
                  std::next(MachineBasicBlock::iterator(MI)), BB->end());
  DoneMBB->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(LoopMBB);

  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  Register ReadAgainReg = RegInfo.createVirtualRegister(&Primate::GPRRegClass);
  Register LoReg = MI.getOperand(0).getReg();
  Register HiReg = MI.getOperand(1).getReg();
  DebugLoc DL = MI.getDebugLoc();

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  BuildMI(LoopMBB, DL, TII->get(Primate::CSRRS), HiReg)
      .addImm(PrimateSysReg::lookupSysRegByName("CYCLEH")->Encoding)
      .addReg(Primate::X0);
  BuildMI(LoopMBB, DL, TII->get(Primate::CSRRS), LoReg)
      .addImm(PrimateSysReg::lookupSysRegByName("CYCLE")->Encoding)
      .addReg(Primate::X0);
  BuildMI(LoopMBB, DL, TII->get(Primate::CSRRS), ReadAgainReg)
      .addImm(PrimateSysReg::lookupSysRegByName("CYCLEH")->Encoding)
      .addReg(Primate::X0);

  BuildMI(LoopMBB, DL, TII->get(Primate::BNE))
      .addReg(HiReg)
      .addReg(ReadAgainReg)
      .addMBB(LoopMBB);

  LoopMBB->addSuccessor(LoopMBB);
  LoopMBB->addSuccessor(DoneMBB);

  MI.eraseFromParent();

  return DoneMBB;
}

static MachineBasicBlock *emitSplitF64Pseudo(MachineInstr &MI,
                                             MachineBasicBlock *BB) {
  assert(MI.getOpcode() == Primate::SplitF64Pseudo && "Unexpected instruction");

  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  Register LoReg = MI.getOperand(0).getReg();
  Register HiReg = MI.getOperand(1).getReg();
  Register SrcReg = MI.getOperand(2).getReg();
  const TargetRegisterClass *SrcRC = &Primate::FPR64RegClass;
  int FI = MF.getInfo<PrimateMachineFunctionInfo>()->getMoveF64FrameIndex(MF);

  TII.storeRegToStackSlot(*BB, MI, SrcReg, MI.getOperand(2).isKill(), FI, SrcRC,
                          RI, Register());
  MachinePointerInfo MPI = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMOLo =
      MF.getMachineMemOperand(MPI, MachineMemOperand::MOLoad, 4, Align(8));
  MachineMemOperand *MMOHi = MF.getMachineMemOperand(
      MPI.getWithOffset(4), MachineMemOperand::MOLoad, 4, Align(8));
  BuildMI(*BB, MI, DL, TII.get(Primate::LW), LoReg)
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMOLo);
  BuildMI(*BB, MI, DL, TII.get(Primate::LW), HiReg)
      .addFrameIndex(FI)
      .addImm(4)
      .addMemOperand(MMOHi);
  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *emitBuildPairF64Pseudo(MachineInstr &MI,
                                                 MachineBasicBlock *BB) {
  assert(MI.getOpcode() == Primate::BuildPairF64Pseudo &&
         "Unexpected instruction");

  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  Register DstReg = MI.getOperand(0).getReg();
  Register LoReg = MI.getOperand(1).getReg();
  Register HiReg = MI.getOperand(2).getReg();
  const TargetRegisterClass *DstRC = &Primate::FPR64RegClass;
  int FI = MF.getInfo<PrimateMachineFunctionInfo>()->getMoveF64FrameIndex(MF);

  MachinePointerInfo MPI = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMOLo =
      MF.getMachineMemOperand(MPI, MachineMemOperand::MOStore, 4, Align(8));
  MachineMemOperand *MMOHi = MF.getMachineMemOperand(
      MPI.getWithOffset(4), MachineMemOperand::MOStore, 4, Align(8));
  BuildMI(*BB, MI, DL, TII.get(Primate::SW))
      .addReg(LoReg, getKillRegState(MI.getOperand(1).isKill()))
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMOLo);
  BuildMI(*BB, MI, DL, TII.get(Primate::SW))
      .addReg(HiReg, getKillRegState(MI.getOperand(2).isKill()))
      .addFrameIndex(FI)
      .addImm(4)
      .addMemOperand(MMOHi);
  TII.loadRegFromStackSlot(*BB, MI, DstReg, FI, DstRC, RI, Register());
  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static bool isSelectPseudo(MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case Primate::Select_GPR_Using_CC_GPR:
  // case Primate::Select_FPR16_Using_CC_GPR:
  // case Primate::Select_FPR32_Using_CC_GPR:
  // case Primate::Select_FPR64_Using_CC_GPR:
    return true;
  }
}

static MachineBasicBlock *emitSelectPseudo(MachineInstr &MI,
                                           MachineBasicBlock *BB) {
  // To "insert" Select_* instructions, we actually have to insert the triangle
  // control-flow pattern.  The incoming instructions know the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and the condcode to use to select the appropriate branch.
  //
  // We produce the following control flow:
  //     HeadMBB
  //     |  \
  //     |  IfFalseMBB
  //     | /
  //    TailMBB
  //
  // When we find a sequence of selects we attempt to optimize their emission
  // by sharing the control flow. Currently we only handle cases where we have
  // multiple selects with the exact same condition (same LHS, RHS and CC).
  // The selects may be interleaved with other instructions if the other
  // instructions meet some requirements we deem safe:
  // - They are debug instructions. Otherwise,
  // - They do not have side-effects, do not access memory and their inputs do
  //   not depend on the results of the select pseudo-instructions.
  // The TrueV/FalseV operands of the selects cannot depend on the result of
  // previous selects in the sequence.
  // These conditions could be further relaxed. See the X86 target for a
  // related approach and more information.
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  auto CC = static_cast<ISD::CondCode>(MI.getOperand(3).getImm());

  SmallVector<MachineInstr *, 4> SelectDebugValues;
  SmallSet<Register, 4> SelectDests;
  SelectDests.insert(MI.getOperand(0).getReg());

  MachineInstr *LastSelectPseudo = &MI;

  for (auto E = BB->end(), SequenceMBBI = MachineBasicBlock::iterator(MI);
       SequenceMBBI != E; ++SequenceMBBI) {
    if (SequenceMBBI->isDebugInstr())
      continue;
    else if (isSelectPseudo(*SequenceMBBI)) {
      if (SequenceMBBI->getOperand(1).getReg() != LHS ||
          SequenceMBBI->getOperand(2).getReg() != RHS ||
          SequenceMBBI->getOperand(3).getImm() != CC ||
          SelectDests.count(SequenceMBBI->getOperand(4).getReg()) ||
          SelectDests.count(SequenceMBBI->getOperand(5).getReg()))
        break;
      LastSelectPseudo = &*SequenceMBBI;
      SequenceMBBI->collectDebugValues(SelectDebugValues);
      SelectDests.insert(SequenceMBBI->getOperand(0).getReg());
    } else {
      if (SequenceMBBI->hasUnmodeledSideEffects() ||
          SequenceMBBI->mayLoadOrStore())
        break;
      if (llvm::any_of(SequenceMBBI->operands(), [&](MachineOperand &MO) {
            return MO.isReg() && MO.isUse() && SelectDests.count(MO.getReg());
          }))
        break;
    }
  }

  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction::iterator I = ++BB->getIterator();

  MachineBasicBlock *HeadMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *TailMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *IfFalseMBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(I, IfFalseMBB);
  F->insert(I, TailMBB);

  // Transfer debug instructions associated with the selects to TailMBB.
  for (MachineInstr *DebugInstr : SelectDebugValues) {
    TailMBB->push_back(DebugInstr->removeFromParent());
  }

  // Move all instructions after the sequence to TailMBB.
  TailMBB->splice(TailMBB->end(), HeadMBB,
                  std::next(LastSelectPseudo->getIterator()), HeadMBB->end());
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi nodes for the selects.
  TailMBB->transferSuccessorsAndUpdatePHIs(HeadMBB);
  // Set the successors for HeadMBB.
  HeadMBB->addSuccessor(IfFalseMBB);
  HeadMBB->addSuccessor(TailMBB);

  // Insert appropriate branch.
  unsigned Opcode = getBranchOpcodeForIntCondCode(CC);

  BuildMI(HeadMBB, DL, TII.get(Opcode))
    .addReg(LHS)
    .addReg(RHS)
    .addMBB(TailMBB);

  // IfFalseMBB just falls through to TailMBB.
  IfFalseMBB->addSuccessor(TailMBB);

  // Create PHIs for all of the select pseudo-instructions.
  auto SelectMBBI = MI.getIterator();
  auto SelectEnd = std::next(LastSelectPseudo->getIterator());
  auto InsertionPoint = TailMBB->begin();
  while (SelectMBBI != SelectEnd) {
    auto Next = std::next(SelectMBBI);
    if (isSelectPseudo(*SelectMBBI)) {
      // %Result = phi [ %TrueValue, HeadMBB ], [ %FalseValue, IfFalseMBB ]
      BuildMI(*TailMBB, InsertionPoint, SelectMBBI->getDebugLoc(),
              TII.get(Primate::PHI), SelectMBBI->getOperand(0).getReg())
          .addReg(SelectMBBI->getOperand(4).getReg())
          .addMBB(HeadMBB)
          .addReg(SelectMBBI->getOperand(5).getReg())
          .addMBB(IfFalseMBB);
      SelectMBBI->eraseFromParent();
    }
    SelectMBBI = Next;
  }

  F->getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
  return TailMBB;
}

MachineBasicBlock *
PrimateTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                 MachineBasicBlock *BB) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case Primate::ReadCycleWide:
    assert(!Subtarget.is64Bit() &&
           "ReadCycleWrite is only to be used on primate32");
    return emitReadCycleWidePseudo(MI, BB);
  case Primate::Select_GPR_Using_CC_GPR:
  // case Primate::Select_FPR16_Using_CC_GPR:
  // case Primate::Select_FPR32_Using_CC_GPR:
  // case Primate::Select_FPR64_Using_CC_GPR:
    return emitSelectPseudo(MI, BB);
  case Primate::BuildPairF64Pseudo:
    return emitBuildPairF64Pseudo(MI, BB);
  case Primate::SplitF64Pseudo:
    return emitSplitF64Pseudo(MI, BB);
  }
}

// Calling Convention Implementation.
// The expectations for frontend ABI lowering vary from target to target.
// Ideally, an LLVM frontend would be able to avoid worrying about many ABI
// details, but this is a longer term goal. For now, we simply try to keep the
// role of the frontend as simple and well-defined as possible. The rules can
// be summarised as:
// * Never split up large scalar arguments. We handle them here.
// * If a hardfloat calling convention is being used, and the struct may be
// passed in a pair of registers (fp+fp, int+fp), and both registers are
// available, then pass as two separate arguments. If either the GPRs or FPRs
// are exhausted, then pass according to the rule below.
// * If a struct could never be passed in registers or directly in a stack
// slot (as it is larger than 2*XLEN and the floating point rules don't
// apply), then pass it using a pointer with the byval attribute.
// * If a struct is less than 2*XLEN, then coerce to either a two-element
// word-sized array or a 2*XLEN scalar (depending on alignment).
// * The frontend can determine whether a struct is returned by reference or
// not based on its size and fields. If it will be returned by reference, the
// frontend must modify the prototype so a pointer with the sret annotation is
// passed as the first argument. This is not necessary for large scalar
// returns.
// * Struct return values and varargs should be coerced to structs containing
// register-size fields in the same situations they would be for fixed
// arguments.

static const MCPhysReg ArgGPRs[] = {
  Primate::X10, Primate::X11, Primate::X12, Primate::X13,
  Primate::X14, Primate::X15, Primate::X16, Primate::X17
};
static const MCPhysReg ArgFPR16s[] = {
  Primate::F10_H, Primate::F11_H, Primate::F12_H, Primate::F13_H,
  Primate::F14_H, Primate::F15_H, Primate::F16_H, Primate::F17_H
};
static const MCPhysReg ArgFPR32s[] = {
  Primate::F10_F, Primate::F11_F, Primate::F12_F, Primate::F13_F,
  Primate::F14_F, Primate::F15_F, Primate::F16_F, Primate::F17_F
};
static const MCPhysReg ArgFPR64s[] = {
  Primate::F10_D, Primate::F11_D, Primate::F12_D, Primate::F13_D,
  Primate::F14_D, Primate::F15_D, Primate::F16_D, Primate::F17_D
};
// This is an interim calling convention and it may be changed in the future.
static const MCPhysReg ArgVRs[] = {
    Primate::V8,  Primate::V9,  Primate::V10, Primate::V11, Primate::V12, Primate::V13,
    Primate::V14, Primate::V15, Primate::V16, Primate::V17, Primate::V18, Primate::V19,
    Primate::V20, Primate::V21, Primate::V22, Primate::V23};
static const MCPhysReg ArgVRM2s[] = {Primate::V8M2,  Primate::V10M2, Primate::V12M2,
                                     Primate::V14M2, Primate::V16M2, Primate::V18M2,
                                     Primate::V20M2, Primate::V22M2};
static const MCPhysReg ArgVRM4s[] = {Primate::V8M4, Primate::V12M4, Primate::V16M4,
                                     Primate::V20M4};
static const MCPhysReg ArgVRM8s[] = {Primate::V8M8, Primate::V16M8};

// Pass a 2*XLEN argument that has been split into two XLEN values through
// registers or the stack as necessary.
static bool CC_PrimateAssign2XLen(unsigned XLen, CCState &State, CCValAssign VA1,
                                ISD::ArgFlagsTy ArgFlags1, unsigned ValNo2,
                                MVT ValVT2, MVT LocVT2,
                                ISD::ArgFlagsTy ArgFlags2) {
  unsigned XLenInBytes = XLen / 8;
  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // At least one half can be passed via register.
    State.addLoc(CCValAssign::getReg(VA1.getValNo(), VA1.getValVT(), Reg,
                                     VA1.getLocVT(), CCValAssign::Full));
  } else {
    // Both halves must be passed on the stack, with proper alignment.
    Align StackAlign =
        std::max(Align(XLenInBytes), ArgFlags1.getNonZeroOrigAlign());
    State.addLoc(
        CCValAssign::getMem(VA1.getValNo(), VA1.getValVT(),
                            State.AllocateStack(XLenInBytes, StackAlign),
                            VA1.getLocVT(), CCValAssign::Full));
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
    return false;
  }

  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // The second half can also be passed via register.
    State.addLoc(
        CCValAssign::getReg(ValNo2, ValVT2, Reg, LocVT2, CCValAssign::Full));
  } else {
    // The second half is passed via the stack, without additional alignment.
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
  }

  return false;
}

static unsigned allocatePRVReg(MVT ValVT, unsigned ValNo,
                               std::optional<unsigned> FirstMaskArgument,
                               CCState &State, const PrimateTargetLowering &TLI) {
  const TargetRegisterClass *RC = TLI.getRegClassFor(ValVT);
  if (RC == &Primate::VRRegClass) {
    // Assign the first mask argument to V0.
    // This is an interim calling convention and it may be changed in the
    // future.
    if (FirstMaskArgument && ValNo == *FirstMaskArgument)
      return State.AllocateReg(Primate::V0);
    return State.AllocateReg(ArgVRs);
  }
  if (RC == &Primate::VRM2RegClass)
    return State.AllocateReg(ArgVRM2s);
  if (RC == &Primate::VRM4RegClass)
    return State.AllocateReg(ArgVRM4s);
  if (RC == &Primate::VRM8RegClass)
    return State.AllocateReg(ArgVRM8s);
  llvm_unreachable("Unhandled register class for ValueType");
}

// Implements the Primate calling convention. Returns true upon failure.
static bool CC_Primate(const DataLayout &DL, PrimateABI::ABI ABI, unsigned ValNo,
                     MVT ValVT, MVT LocVT, CCValAssign::LocInfo LocInfo,
                     ISD::ArgFlagsTy ArgFlags, CCState &State, bool IsFixed,
                     bool IsRet, Type *OrigTy, const PrimateTargetLowering &TLI,
                     std::optional<unsigned> FirstMaskArgument) {
  unsigned XLen = DL.getLargestLegalIntTypeSizeInBits();
  assert(XLen == 32 || XLen == 64);
  MVT XLenVT = XLen == 32 ? MVT::i32 : MVT::i64;

  // Any return value split in to more than two values can't be returned
  // directly. Vectors are returned via the available vector registers.
  if (!LocVT.isVector() && IsRet && ValNo > 1)
    return true;

  // UseGPRForF16_F32 if targeting one of the soft-float ABIs, if passing a
  // variadic argument, or if no F16/F32 argument registers are available.
  bool UseGPRForF16_F32 = true;
  // UseGPRForF64 if targeting soft-float ABIs or an FLEN=32 ABI, if passing a
  // variadic argument, or if no F64 argument registers are available.
  bool UseGPRForF64 = true;

  switch (ABI) {
  default:
    llvm_unreachable("Unexpected ABI");
  case PrimateABI::ABI_ILP32:
  case PrimateABI::ABI_LP64:
    break;
  case PrimateABI::ABI_ILP32F:
  case PrimateABI::ABI_LP64F:
    UseGPRForF16_F32 = !IsFixed;
    break;
  case PrimateABI::ABI_ILP32D:
  case PrimateABI::ABI_LP64D:
    UseGPRForF16_F32 = !IsFixed;
    UseGPRForF64 = !IsFixed;
    break;
  }

  // FPR16, FPR32, and FPR64 alias each other.
  if (State.getFirstUnallocated(ArgFPR32s) == std::size(ArgFPR32s)) {
    UseGPRForF16_F32 = true;
    UseGPRForF64 = true;
  }

  // From this point on, rely on UseGPRForF16_F32, UseGPRForF64 and
  // similar local variables rather than directly checking against the target
  // ABI.

  if (UseGPRForF16_F32 && (ValVT == MVT::f16 || ValVT == MVT::f32)) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::BCvt;
  } else if (UseGPRForF64 && XLen == 64 && ValVT == MVT::f64) {
    LocVT = MVT::i64;
    LocInfo = CCValAssign::BCvt;
  }

  // If this is a variadic argument, the Primate calling convention requires
  // that it is assigned an 'even' or 'aligned' register if it has 8-byte
  // alignment (PR32) or 16-byte alignment (PR64). An aligned register should
  // be used regardless of whether the original argument was split during
  // legalisation or not. The argument will not be passed by registers if the
  // original type is larger than 2*XLEN, so the register alignment rule does
  // not apply.
  unsigned TwoXLenInBytes = (2 * XLen) / 8;
  if (!IsFixed && ArgFlags.getNonZeroOrigAlign() == TwoXLenInBytes &&
      DL.getTypeAllocSize(OrigTy) == TwoXLenInBytes) {
    unsigned RegIdx = State.getFirstUnallocated(ArgGPRs);
    // Skip 'odd' register if necessary.
    if (RegIdx != std::size(ArgGPRs) && RegIdx % 2 == 1)
      State.AllocateReg(ArgGPRs);
  }

  SmallVectorImpl<CCValAssign> &PendingLocs = State.getPendingLocs();
  SmallVectorImpl<ISD::ArgFlagsTy> &PendingArgFlags =
      State.getPendingArgFlags();

  assert(PendingLocs.size() == PendingArgFlags.size() &&
         "PendingLocs and PendingArgFlags out of sync");

  // Handle passing f64 on PR32D with a soft float ABI or when floating point
  // registers are exhausted.
  if (UseGPRForF64 && XLen == 32 && ValVT == MVT::f64) {
    assert(!ArgFlags.isSplit() && PendingLocs.empty() &&
           "Can't lower f64 if it is split");
    // Depending on available argument GPRS, f64 may be passed in a pair of
    // GPRs, split between a GPR and the stack, or passed completely on the
    // stack. LowerCall/LowerFormalArguments/LowerReturn must recognise these
    // cases.
    Register Reg = State.AllocateReg(ArgGPRs);
    LocVT = MVT::i32;
    if (!Reg) {
      unsigned StackOffset = State.AllocateStack(8, Align(8));
      State.addLoc(
          CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      return false;
    }
    if (!State.AllocateReg(ArgGPRs))
      State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // Fixed-length vectors are located in the corresponding scalable-vector
  // container types.
  if (ValVT.isFixedLengthVector())
    LocVT = TLI.getContainerForFixedLengthVector(LocVT);

  // Split arguments might be passed indirectly, so keep track of the pending
  // values. Split vectors are passed via a mix of registers and indirectly, so
  // treat them as we would any other argument.
  if (ValVT.isScalarInteger() && (ArgFlags.isSplit() || !PendingLocs.empty())) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::Indirect;
    PendingLocs.push_back(
        CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));
    PendingArgFlags.push_back(ArgFlags);
    if (!ArgFlags.isSplitEnd()) {
      return false;
    }
  }

  // If the split argument only had two elements, it should be passed directly
  // in registers or on the stack.
  if (ValVT.isScalarInteger() && ArgFlags.isSplitEnd() &&
      PendingLocs.size() <= 2) {
    assert(PendingLocs.size() == 2 && "Unexpected PendingLocs.size()");
    // Apply the normal calling convention rules to the first half of the
    // split argument.
    CCValAssign VA = PendingLocs[0];
    ISD::ArgFlagsTy AF = PendingArgFlags[0];
    PendingLocs.clear();
    PendingArgFlags.clear();
    return CC_PrimateAssign2XLen(XLen, State, VA, AF, ValNo, ValVT, LocVT,
                               ArgFlags);
  }

  // Allocate to a register if possible, or else a stack slot.
  Register Reg;
  unsigned StoreSizeBytes = XLen / 8;
  Align StackAlign = Align(XLen / 8);

  if (ValVT == MVT::f16 && !UseGPRForF16_F32)
    Reg = State.AllocateReg(ArgFPR16s);
  else if (ValVT == MVT::f32 && !UseGPRForF16_F32)
    Reg = State.AllocateReg(ArgFPR32s);
  else if (ValVT == MVT::f64 && !UseGPRForF64)
    Reg = State.AllocateReg(ArgFPR64s);
  else if (ValVT.isVector()) {
    Reg = allocatePRVReg(ValVT, ValNo, FirstMaskArgument, State, TLI);
    if (!Reg) {
      // For return values, the vector must be passed fully via registers or
      // via the stack.
      // FIXME: The proposed vector ABI only mandates v8-v15 for return values,
      // but we're using all of them.
      if (IsRet)
        return true;
      // Try using a GPR to pass the address
      if ((Reg = State.AllocateReg(ArgGPRs))) {
        LocVT = XLenVT;
        LocInfo = CCValAssign::Indirect;
      } else if (ValVT.isScalableVector()) {
        report_fatal_error("Unable to pass scalable vector types on the stack");
      } else {
        // Pass fixed-length vectors on the stack.
        LocVT = ValVT;
        StoreSizeBytes = ValVT.getStoreSize();
        // Align vectors to their element sizes, being careful for vXi1
        // vectors.
        StackAlign = MaybeAlign(ValVT.getScalarSizeInBits() / 8).valueOrOne();
      }
    }
  } else {
    Reg = State.AllocateReg(ArgGPRs);
  }

  unsigned StackOffset =
      Reg ? 0 : State.AllocateStack(StoreSizeBytes, StackAlign);

  // If we reach this point and PendingLocs is non-empty, we must be at the
  // end of a split argument that must be passed indirectly.
  if (!PendingLocs.empty()) {
    assert(ArgFlags.isSplitEnd() && "Expected ArgFlags.isSplitEnd()");
    assert(PendingLocs.size() > 2 && "Unexpected PendingLocs.size()");

    for (auto &It : PendingLocs) {
      if (Reg)
        It.convertToReg(Reg);
      else
        It.convertToMem(StackOffset);
      State.addLoc(It);
    }
    PendingLocs.clear();
    PendingArgFlags.clear();
    return false;
  }

  assert((!UseGPRForF16_F32 || !UseGPRForF64 || LocVT == XLenVT ||
          (TLI.getSubtarget().hasStdExtV() && ValVT.isVector())) &&
         "Expected an XLenVT or vector types at this stage");

  if (Reg) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // When a floating-point value is passed on the stack, no bit-conversion is
  // needed.
  if (ValVT.isFloatingPoint()) {
    LocVT = ValVT;
    LocInfo = CCValAssign::Full;
  }
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
  return false;
}

template <typename ArgTy>
static std::optional<unsigned> preAssignMask(const ArgTy &Args) {
  for (const auto &ArgIdx : enumerate(Args)) {
    MVT ArgVT = ArgIdx.value().VT;
    if (ArgVT.isVector() && ArgVT.getVectorElementType() == MVT::i1)
      return ArgIdx.index();
  }
  return {};
}

void PrimateTargetLowering::analyzeInputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet,
    PrimateCCAssignFn Fn) const {
  unsigned NumArgs = Ins.size();
  FunctionType *FType = MF.getFunction().getFunctionType();

  std::optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasStdExtV())
    FirstMaskArgument = preAssignMask(Ins);

  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ArgVT = Ins[i].VT;
    ISD::ArgFlagsTy ArgFlags = Ins[i].Flags;

    Type *ArgTy = nullptr;
    if (IsRet)
      ArgTy = FType->getReturnType();
    else if (Ins[i].isOrigArg())
      ArgTy = FType->getParamType(Ins[i].getOrigArgIndex());

    PrimateABI::ABI ABI = MF.getSubtarget<PrimateSubtarget>().getTargetABI();
    if (Fn(MF.getDataLayout(), ABI, i, ArgVT, ArgVT, CCValAssign::Full,
           ArgFlags, CCInfo, /*IsFixed=*/true, IsRet, ArgTy, *this,
           FirstMaskArgument)) {
      LLVM_DEBUG(dbgs() << "InputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << '\n');
      llvm_unreachable(nullptr);
    }
  }
}

void PrimateTargetLowering::analyzeOutputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::OutputArg> &Outs, bool IsRet,
    CallLoweringInfo *CLI, PrimateCCAssignFn Fn) const {
  unsigned NumArgs = Outs.size();

  std::optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasStdExtV())
    FirstMaskArgument = preAssignMask(Outs);

  for (unsigned i = 0; i != NumArgs; i++) {
    MVT ArgVT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    Type *OrigTy = CLI ? CLI->getArgs()[Outs[i].OrigArgIndex].Ty : nullptr;

    PrimateABI::ABI ABI = MF.getSubtarget<PrimateSubtarget>().getTargetABI();
    if (Fn(MF.getDataLayout(), ABI, i, ArgVT, ArgVT, CCValAssign::Full,
           ArgFlags, CCInfo, Outs[i].IsFixed, IsRet, OrigTy, *this,
           FirstMaskArgument)) {
      LLVM_DEBUG(dbgs() << "OutputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << "\n");
      llvm_unreachable(nullptr);
    }
  }
}

// Convert Val to a ValVT. Should not be called for CCValAssign::Indirect
// values.
static SDValue convertLocVTToValVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL,
                                   const PrimateSubtarget &Subtarget) {
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    if (VA.getValVT().isFixedLengthVector() && VA.getLocVT().isScalableVector())
      Val = convertFromScalableVector(VA.getValVT(), Val, DAG, Subtarget);
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT().isInteger() && VA.getValVT() == MVT::f16)
      Val = DAG.getNode(PrimateISD::FMV_H_X, DL, MVT::f16, Val);
    else if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32)
      Val = DAG.getNode(PrimateISD::FMV_W_X_PR64, DL, MVT::f32, Val);
    else
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromRegLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL,
                                const PrimateTargetLowering &TLI) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  EVT LocVT = VA.getLocVT();
  SDValue Val;
  const TargetRegisterClass *RC = TLI.getRegClassFor(LocVT.getSimpleVT());
  Register VReg = RegInfo.createVirtualRegister(RC);
  RegInfo.addLiveIn(VA.getLocReg(), VReg);
  Val = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);

  if (VA.getLocInfo() == CCValAssign::Indirect)
    return Val;

  return convertLocVTToValVT(DAG, Val, VA, DL, TLI.getSubtarget());
}

static SDValue convertValVTToLocVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL,
                                   const PrimateSubtarget &Subtarget) {
  EVT LocVT = VA.getLocVT();

  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    if (VA.getValVT().isFixedLengthVector() && LocVT.isScalableVector())
      Val = convertToScalableVector(LocVT, Val, DAG, Subtarget);
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT().isInteger() && VA.getValVT() == MVT::f16)
      Val = DAG.getNode(PrimateISD::FMV_X_ANYEXTH, DL, VA.getLocVT(), Val);
    else if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32)
      Val = DAG.getNode(PrimateISD::FMV_X_ANYEXTW_PR64, DL, MVT::i64, Val);
    else
      Val = DAG.getNode(ISD::BITCAST, DL, LocVT, Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromMemLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  EVT LocVT = VA.getLocVT();
  EVT ValVT = VA.getValVT();
  EVT PtrVT = MVT::getIntegerVT(DAG.getDataLayout().getPointerSizeInBits(0));
  int FI = MFI.CreateFixedObject(ValVT.getStoreSize(), VA.getLocMemOffset(),
                                 /*Immutable=*/true);
  SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
  SDValue Val;

  ISD::LoadExtType ExtType;
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
  case CCValAssign::Indirect:
  case CCValAssign::BCvt:
    ExtType = ISD::NON_EXTLOAD;
    break;
  }
  Val = DAG.getExtLoad(
      ExtType, DL, LocVT, Chain, FIN,
      MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI), ValVT);
  return Val;
}

static SDValue unpackF64OnPR32DSoftABI(SelectionDAG &DAG, SDValue Chain,
                                       const CCValAssign &VA, const SDLoc &DL) {
  assert(VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64 &&
         "Unexpected VA");
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  if (VA.isMemLoc()) {
    // f64 is passed on the stack.
    int FI = MFI.CreateFixedObject(8, VA.getLocMemOffset(), /*Immutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    return DAG.getLoad(MVT::f64, DL, Chain, FIN,
                       MachinePointerInfo::getFixedStack(MF, FI));
  }

  assert(VA.isRegLoc() && "Expected register VA assignment");

  Register LoVReg = RegInfo.createVirtualRegister(&Primate::GPRRegClass);
  RegInfo.addLiveIn(VA.getLocReg(), LoVReg);
  SDValue Lo = DAG.getCopyFromReg(Chain, DL, LoVReg, MVT::i32);
  SDValue Hi;
  if (VA.getLocReg() == Primate::X17) {
    // Second half of f64 is passed on the stack.
    int FI = MFI.CreateFixedObject(4, 0, /*Immutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    Hi = DAG.getLoad(MVT::i32, DL, Chain, FIN,
                     MachinePointerInfo::getFixedStack(MF, FI));
  } else {
    // Second half of f64 is passed in another GPR.
    Register HiVReg = RegInfo.createVirtualRegister(&Primate::GPRRegClass);
    RegInfo.addLiveIn(VA.getLocReg() + 1, HiVReg);
    Hi = DAG.getCopyFromReg(Chain, DL, HiVReg, MVT::i32);
  }
  return DAG.getNode(PrimateISD::BuildPairF64, DL, MVT::f64, Lo, Hi);
}

// FastCC has less than 1% performance improvement for some particular
// benchmark. But theoretically, it may has benenfit for some cases.
static bool CC_Primate_FastCC(const DataLayout &DL, PrimateABI::ABI ABI,
                            unsigned ValNo, MVT ValVT, MVT LocVT,
                            CCValAssign::LocInfo LocInfo,
                            ISD::ArgFlagsTy ArgFlags, CCState &State,
                            bool IsFixed, bool IsRet, Type *OrigTy,
                            const PrimateTargetLowering &TLI,
                            std::optional<unsigned> FirstMaskArgument) {

  // X5 and X6 might be used for save-restore libcall.
  static const MCPhysReg GPRList[] = {
      Primate::X10, Primate::X11, Primate::X12, Primate::X13, Primate::X14,
      Primate::X15, Primate::X16, Primate::X17, Primate::X7,  Primate::X28,
      Primate::X29, Primate::X30, Primate::X31};

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    if (unsigned Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f16) {
    static const MCPhysReg FPR16List[] = {
        Primate::F10_H, Primate::F11_H, Primate::F12_H, Primate::F13_H, Primate::F14_H,
        Primate::F15_H, Primate::F16_H, Primate::F17_H, Primate::F0_H,  Primate::F1_H,
        Primate::F2_H,  Primate::F3_H,  Primate::F4_H,  Primate::F5_H,  Primate::F6_H,
        Primate::F7_H,  Primate::F28_H, Primate::F29_H, Primate::F30_H, Primate::F31_H};
    if (unsigned Reg = State.AllocateReg(FPR16List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32) {
    static const MCPhysReg FPR32List[] = {
        Primate::F10_F, Primate::F11_F, Primate::F12_F, Primate::F13_F, Primate::F14_F,
        Primate::F15_F, Primate::F16_F, Primate::F17_F, Primate::F0_F,  Primate::F1_F,
        Primate::F2_F,  Primate::F3_F,  Primate::F4_F,  Primate::F5_F,  Primate::F6_F,
        Primate::F7_F,  Primate::F28_F, Primate::F29_F, Primate::F30_F, Primate::F31_F};
    if (unsigned Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64) {
    static const MCPhysReg FPR64List[] = {
        Primate::F10_D, Primate::F11_D, Primate::F12_D, Primate::F13_D, Primate::F14_D,
        Primate::F15_D, Primate::F16_D, Primate::F17_D, Primate::F0_D,  Primate::F1_D,
        Primate::F2_D,  Primate::F3_D,  Primate::F4_D,  Primate::F5_D,  Primate::F6_D,
        Primate::F7_D,  Primate::F28_D, Primate::F29_D, Primate::F30_D, Primate::F31_D};
    if (unsigned Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    unsigned Offset4 = State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset4, LocVT, LocInfo));
    return false;
  }

  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    unsigned Offset5 = State.AllocateStack(8, Align(8));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset5, LocVT, LocInfo));
    return false;
  }

  if (LocVT.isVector()) {
    if (unsigned Reg =
            allocatePRVReg(ValVT, ValNo, FirstMaskArgument, State, TLI)) {
      // Fixed-length vectors are located in the corresponding scalable-vector
      // container types.
      if (ValVT.isFixedLengthVector())
        LocVT = TLI.getContainerForFixedLengthVector(LocVT);
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    } else {
      // Try and pass the address via a "fast" GPR.
      if (unsigned GPRReg = State.AllocateReg(GPRList)) {
        LocInfo = CCValAssign::Indirect;
        LocVT = TLI.getSubtarget().getXLenVT();
        State.addLoc(CCValAssign::getReg(ValNo, ValVT, GPRReg, LocVT, LocInfo));
      } else if (ValVT.isFixedLengthVector()) {
        auto StackAlign =
            MaybeAlign(ValVT.getScalarSizeInBits() / 8).valueOrOne();
        unsigned StackOffset =
            State.AllocateStack(ValVT.getStoreSize(), StackAlign);
        State.addLoc(
            CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      } else {
        // Can't pass scalable vectors on the stack.
        return true;
      }
    }

    return false;
  }

  return true; // CC didn't match.
}

static bool CC_Primate_GHC(unsigned ValNo, MVT ValVT, MVT LocVT,
                         CCValAssign::LocInfo LocInfo,
                         ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    // Pass in STG registers: Base, Sp, Hp, R1, R2, R3, R4, R5, R6, R7, SpLim
    //                        s1    s2  s3  s4  s5  s6  s7  s8  s9  s10 s11
    static const MCPhysReg GPRList[] = {
        Primate::X9, Primate::X18, Primate::X19, Primate::X20, Primate::X21, Primate::X22,
        Primate::X23, Primate::X24, Primate::X25, Primate::X26, Primate::X27};
    if (unsigned Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32) {
    // Pass in STG registers: F1, ..., F6
    //                        fs0 ... fs5
    static const MCPhysReg FPR32List[] = {Primate::F8_F, Primate::F9_F,
                                          Primate::F18_F, Primate::F19_F,
                                          Primate::F20_F, Primate::F21_F};
    if (unsigned Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64) {
    // Pass in STG registers: D1, ..., D6
    //                        fs6 ... fs11
    static const MCPhysReg FPR64List[] = {Primate::F22_D, Primate::F23_D,
                                          Primate::F24_D, Primate::F25_D,
                                          Primate::F26_D, Primate::F27_D};
    if (unsigned Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  report_fatal_error("No registers left in GHC calling convention");
  return true;
}

// Transform physical registers into virtual registers.
SDValue PrimateTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  MachineFunction &MF = DAG.getMachineFunction();

  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  case CallingConv::GHC:
    if (!MF.getSubtarget().getFeatureBits()[Primate::FeatureStdExtF] ||
        !MF.getSubtarget().getFeatureBits()[Primate::FeatureStdExtD])
      report_fatal_error(
        "GHC calling convention requires the F and D instruction set extensions");
  }

  const Function &Func = MF.getFunction();
  if (Func.hasFnAttribute("interrupt")) {
    if (!Func.arg_empty())
      report_fatal_error(
        "Functions with the interrupt attribute cannot have arguments!");

    StringRef Kind =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

    if (!(Kind == "user" || Kind == "supervisor" || Kind == "machine"))
      report_fatal_error(
        "Function interrupt attribute argument not supported!");
  }

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = Subtarget.getXLenVT();
  unsigned XLenInBytes = Subtarget.getXLen() / 8;
  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (CallConv == CallingConv::GHC)
    CCInfo.AnalyzeFormalArguments(Ins, CC_Primate_GHC);
  else
    analyzeInputArgs(MF, CCInfo, Ins, /*IsRet=*/false,
                     CallConv == CallingConv::Fast ? CC_Primate_FastCC
                                                   : CC_Primate);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue;
    // Passing f64 on PR32D with a soft float ABI must be handled as a special
    // case.
    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64)
      ArgValue = unpackF64OnPR32DSoftABI(DAG, Chain, VA, DL);
    else if (VA.isRegLoc())
      ArgValue = unpackFromRegLoc(DAG, Chain, VA, DL, *this);
    else
      ArgValue = unpackFromMemLoc(DAG, Chain, VA, DL);

    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // If the original argument was split and passed by reference (e.g. i128
      // on PR32), we need to load all parts of it here (using the same
      // address). Vectors may be partly split to registers and partly to the
      // stack, in which case the base address is partly offset and subsequent
      // stores are relative to that.
      InVals.push_back(DAG.getLoad(VA.getValVT(), DL, Chain, ArgValue,
                                   MachinePointerInfo()));
      unsigned ArgIndex = Ins[i].OrigArgIndex;
      unsigned ArgPartOffset = Ins[i].PartOffset;
      assert(VA.getValVT().isVector() || ArgPartOffset == 0);
      while (i + 1 != e && Ins[i + 1].OrigArgIndex == ArgIndex) {
        CCValAssign &PartVA = ArgLocs[i + 1];
        unsigned PartOffset = Ins[i + 1].PartOffset - ArgPartOffset;
        SDValue Offset = DAG.getIntPtrConstant(PartOffset, DL);
        if (PartVA.getValVT().isScalableVector())
          Offset = DAG.getNode(ISD::VSCALE, DL, XLenVT, Offset);
        SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, ArgValue, Offset);
        InVals.push_back(DAG.getLoad(PartVA.getValVT(), DL, Chain, Address,
                                     MachinePointerInfo()));
        ++i;
      }
      continue;
    }
    InVals.push_back(ArgValue);
  }

  if (IsVarArg) {
    ArrayRef<MCPhysReg> ArgRegs = ArrayRef(ArgGPRs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &Primate::GPRRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    PrimateMachineFunctionInfo *PRFI = MF.getInfo<PrimateMachineFunctionInfo>();

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a7.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getStackSize();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = XLenInBytes * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
    PRFI->setVarArgsFrameIndex(FI);

    // If saving an odd number of registers then create an extra stack slot to
    // ensure that the frame pointer is 2*XLEN-aligned, which in turn ensures
    // offsets to even-numbered registered remain 2*XLEN-aligned.
    if (Idx % 2) {
      MFI.CreateFixedObject(XLenInBytes, VaArgOffset - (int)XLenInBytes, true);
      VarArgsSaveSize += XLenInBytes;
    }

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size();
         ++I, VaArgOffset += XLenInBytes) {
      const Register Reg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(ArgRegs[I], Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, XLenVT);
      FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
    PRFI->setVarArgsSaveSize(VarArgsSaveSize);
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens for vararg functions.
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

/// isEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization.
/// Note: This is modelled after ARM's IsEligibleForTailCallOptimization.
bool PrimateTargetLowering::isEligibleForTailCallOptimization(
    CCState &CCInfo, CallLoweringInfo &CLI, MachineFunction &MF,
    const SmallVector<CCValAssign, 16> &ArgLocs) const {

  auto &Callee = CLI.Callee;
  auto CalleeCC = CLI.CallConv;
  auto &Outs = CLI.Outs;
  auto &Caller = MF.getFunction();
  auto CallerCC = Caller.getCallingConv();

  // Exception-handling functions need a special set of instructions to
  // indicate a return to the hardware. Tail-calling another function would
  // probably break this.
  // TODO: The "interrupt" attribute isn't currently defined by Primate. This
  // should be expanded as new function attributes are introduced.
  if (Caller.hasFnAttribute("interrupt"))
    return false;

  // Do not tail call opt if the stack is used to pass parameters.
  if (CCInfo.getStackSize() != 0)
    return false;

  // Do not tail call opt if any parameters need to be passed indirectly.
  // Since long doubles (fp128) and i128 are larger than 2*XLEN, they are
  // passed indirectly. So the address of the value will be passed in a
  // register, or if not available, then the address is put on the stack. In
  // order to pass indirectly, space on the stack often needs to be allocated
  // in order to store the value. In this case the CCInfo.getNextStackOffset()
  // != 0 check is not enough and we need to check if any CCValAssign ArgsLocs
  // are passed CCValAssign::Indirect.
  for (auto &VA : ArgLocs)
    if (VA.getLocInfo() == CCValAssign::Indirect)
      return false;

  // Do not tail call opt if either caller or callee uses struct return
  // semantics.
  auto IsCallerStructRet = Caller.hasStructRetAttr();
  auto IsCalleeStructRet = Outs.empty() ? false : Outs[0].Flags.isSRet();
  if (IsCallerStructRet || IsCalleeStructRet)
    return false;

  // Externally-defined functions with weak linkage should not be
  // tail-called. The behaviour of branch instructions in this situation (as
  // used for tail calls) is implementation-defined, so we cannot rely on the
  // linker replacing the tail call with a return.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    if (GV->hasExternalWeakLinkage())
      return false;
  }

  // The callee has to preserve all registers the caller needs to preserve.
  const PrimateRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);
  if (CalleeCC != CallerCC) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }

  // Byval parameters hand the function a pointer directly into the stack area
  // we want to reuse during a tail call. Working around this *is* possible
  // but less efficient and uglier in LowerCall.
  for (auto &Arg : Outs)
    if (Arg.Flags.isByVal())
      return false;

  return true;
}

static Align getPrefTypeAlign(EVT VT, SelectionDAG &DAG) {
  return DAG.getDataLayout().getPrefTypeAlign(
      VT.getTypeForEVT(*DAG.getContext()));
}

// Lower a call to a callseq_start + CALL + callseq_end chain, and add input
// and output parameter nodes.
SDValue PrimateTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                       SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = Subtarget.getXLenVT();

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (CallConv == CallingConv::GHC)
    ArgCCInfo.AnalyzeCallOperands(Outs, CC_Primate_GHC);
  else
    analyzeOutputArgs(MF, ArgCCInfo, Outs, /*IsRet=*/false, &CLI,
                      CallConv == CallingConv::Fast ? CC_Primate_FastCC
                                                    : CC_Primate);

  // Check if it's really possible to do a tail call.
  if (IsTailCall)
    IsTailCall = isEligibleForTailCallOptimization(ArgCCInfo, CLI, MF, ArgLocs);

  if (IsTailCall)
    ++NumTailCalls;
  else if (CLI.CB && CLI.CB->isMustTailCall())
    report_fatal_error("failed to perform tail call elimination on a call "
                       "site marked musttail");

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getStackSize();

  // Create local copies for byval args
  SmallVector<SDValue, 8> ByValArgs;
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    if (!Flags.isByVal())
      continue;

    SDValue Arg = OutVals[i];
    unsigned Size = Flags.getByValSize();
    Align Alignment = Flags.getNonZeroByValAlign();

    int FI =
        MF.getFrameInfo().CreateStackObject(Size, Alignment, /*isSS=*/false);
    SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
    SDValue SizeNode = DAG.getConstant(Size, DL, XLenVT);

    Chain = DAG.getMemcpy(Chain, DL, FIPtr, Arg, SizeNode, Alignment,
                          /*IsVolatile=*/false,
                          /*AlwaysInline=*/false, IsTailCall,
                          MachinePointerInfo(), MachinePointerInfo());
    ByValArgs.push_back(FIPtr);
  }

  if (!IsTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<Register, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned i = 0, j = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // Handle passing f64 on PR32D with a soft float ABI as a special case.
    bool IsF64OnPR32DSoftABI =
        VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64;
    if (IsF64OnPR32DSoftABI && VA.isRegLoc()) {
      SDValue SplitF64 = DAG.getNode(
          PrimateISD::SplitF64, DL, DAG.getVTList(MVT::i32, MVT::i32), ArgValue);
      SDValue Lo = SplitF64.getValue(0);
      SDValue Hi = SplitF64.getValue(1);

      Register RegLo = VA.getLocReg();
      RegsToPass.push_back(std::make_pair(RegLo, Lo));

      if (RegLo == Primate::X17) {
        // Second half of f64 is passed on the stack.
        // Work out the address of the stack slot.
        if (!StackPtr.getNode())
          StackPtr = DAG.getCopyFromReg(Chain, DL, Primate::X2, PtrVT);
        // Emit the store.
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, Hi, StackPtr, MachinePointerInfo()));
      } else {
        // Second half of f64 is passed in another GPR.
        assert(RegLo < Primate::X31 && "Invalid register pair");
        Register RegHigh = RegLo + 1;
        RegsToPass.push_back(std::make_pair(RegHigh, Hi));
      }
      continue;
    }

    // IsF64OnPR32DSoftABI && VA.isMemLoc() is handled below in the same way
    // as any other MemLoc.

    // Promote the value if needed.
    // For now, only handle fully promoted and indirect arguments.
    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // Store the argument in a stack slot and pass its address.
      Align StackAlign =
          std::max(getPrefTypeAlign(Outs[i].ArgVT, DAG),
                   getPrefTypeAlign(ArgValue.getValueType(), DAG));
      TypeSize StoredSize = ArgValue.getValueType().getStoreSize();
      // If the original argument was split (e.g. i128), we need
      // to store the required parts of it here (and pass just one address).
      // Vectors may be partly split to registers and partly to the stack, in
      // which case the base address is partly offset and subsequent stores are
      // relative to that.
      unsigned ArgIndex = Outs[i].OrigArgIndex;
      unsigned ArgPartOffset = Outs[i].PartOffset;
      assert(VA.getValVT().isVector() || ArgPartOffset == 0);
      // Calculate the total size to store. We don't have access to what we're
      // actually storing other than performing the loop and collecting the
      // info.
      SmallVector<std::pair<SDValue, SDValue>> Parts;
      while (i + 1 != e && Outs[i + 1].OrigArgIndex == ArgIndex) {
        SDValue PartValue = OutVals[i + 1];
        unsigned PartOffset = Outs[i + 1].PartOffset - ArgPartOffset;
        SDValue Offset = DAG.getIntPtrConstant(PartOffset, DL);
        EVT PartVT = PartValue.getValueType();
        if (PartVT.isScalableVector())
          Offset = DAG.getNode(ISD::VSCALE, DL, XLenVT, Offset);
        StoredSize += PartVT.getStoreSize();
        StackAlign = std::max(StackAlign, getPrefTypeAlign(PartVT, DAG));
        Parts.push_back(std::make_pair(PartValue, Offset));
        ++i;
      }
      SDValue SpillSlot = DAG.CreateStackTemporary(StoredSize, StackAlign);
      int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, SpillSlot,
                       MachinePointerInfo::getFixedStack(MF, FI)));
      for (const auto &Part : Parts) {
        SDValue PartValue = Part.first;
        SDValue PartOffset = Part.second;
        SDValue Address =
            DAG.getNode(ISD::ADD, DL, PtrVT, SpillSlot, PartOffset);
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, PartValue, Address,
                         MachinePointerInfo::getFixedStack(MF, FI)));
      }
      ArgValue = SpillSlot;
    } else {
      ArgValue = convertValVTToLocVT(DAG, ArgValue, VA, DL, Subtarget);
    }

    // Use local copy if it is a byval arg.
    if (Flags.isByVal())
      ArgValue = ByValArgs[j++];

    if (VA.isRegLoc()) {
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");
      assert(!IsTailCall && "Tail call not allowed if stack is used "
                            "for passing parameters");

      // Work out the address of the stack slot.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, Primate::X2, PtrVT);
      SDValue Address =
          DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                      DAG.getIntPtrConstant(VA.getLocMemOffset(), DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  SDValue Glue;

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, Glue);
    Glue = Chain.getValue(1);
  }

  // Validate that none of the argument registers have been marked as
  // reserved, if so report an error. Do the same for the return address if this
  // is not a tailcall.
  validateCCReservedRegs(RegsToPass, MF);
  if (!IsTailCall &&
      MF.getSubtarget<PrimateSubtarget>().isRegisterReservedByUser(Primate::X1))
    MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(),
        "Return address register required, but has been reserved."});

  // If the callee is a GlobalAddress/ExternalSymbol node, turn it into a
  // TargetGlobalAddress/TargetExternalSymbol node so that legalize won't
  // split it and then direct call can be matched by PseudoCALL.
  if (GlobalAddressSDNode *S = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = S->getGlobal();

    unsigned OpFlags = PrimateII::MO_CALL;
    if (!getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV))
      OpFlags = PrimateII::MO_PLT;

    Callee = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, OpFlags);
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    unsigned OpFlags = PrimateII::MO_CALL;

    if (!getTargetMachine().shouldAssumeDSOLocal(*MF.getFunction().getParent(),
                                                 nullptr))
      OpFlags = PrimateII::MO_PLT;

    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), PtrVT, OpFlags);
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  if (!IsTailCall) {
    // Add a register mask operand representing the call-preserved registers.
    const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
    const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
    assert(Mask && "Missing call preserved mask for calling convention");
    Ops.push_back(DAG.getRegisterMask(Mask));
  }

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  if (IsTailCall) {
    MF.getFrameInfo().setHasTailCall();
    return DAG.getNode(PrimateISD::TAIL, DL, NodeTys, Ops);
  }

  Chain = DAG.getNode(PrimateISD::CALL, DL, NodeTys, Ops);
  DAG.addNoMergeSiteInfo(Chain.getNode(), CLI.NoMerge);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true),
                             Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> PRLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, PRLocs, *DAG.getContext());
  analyzeInputArgs(MF, RetCCInfo, Ins, /*IsRet=*/true, CC_Primate);

  // Copy all of the result registers out of their specified physreg.
  for (auto &VA : PRLocs) {
    // Copy the value out
    SDValue RetValue =
        DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), Glue);
    // Glue the RetValue to the end of the call sequence
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64) {
      assert(VA.getLocReg() == ArgGPRs[0] && "Unexpected reg assignment");
      SDValue RetValue2 =
          DAG.getCopyFromReg(Chain, DL, ArgGPRs[1], MVT::i32, Glue);
      Chain = RetValue2.getValue(1);
      Glue = RetValue2.getValue(2);
      RetValue = DAG.getNode(PrimateISD::BuildPairF64, DL, MVT::f64, RetValue,
                             RetValue2);
    }

    RetValue = convertLocVTToValVT(DAG, RetValue, VA, DL, Subtarget);

    InVals.push_back(RetValue);
  }

  return Chain;
}

bool PrimateTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> PRLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, PRLocs, Context);

  std::optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasStdExtV())
    FirstMaskArgument = preAssignMask(Outs);

  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    MVT VT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    PrimateABI::ABI ABI = MF.getSubtarget<PrimateSubtarget>().getTargetABI();
    if (CC_Primate(MF.getDataLayout(), ABI, i, VT, VT, CCValAssign::Full,
                 ArgFlags, CCInfo, /*IsFixed=*/true, /*IsRet=*/true, nullptr,
                 *this, FirstMaskArgument))
      return false;
  }
  return true;
}

SDValue
PrimateTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                 bool IsVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 const SDLoc &DL, SelectionDAG &DAG) const {
  const MachineFunction &MF = DAG.getMachineFunction();
  const PrimateSubtarget &STI = MF.getSubtarget<PrimateSubtarget>();

  // Stores the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> PRLocs;

  // Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), PRLocs,
                 *DAG.getContext());

  analyzeOutputArgs(DAG.getMachineFunction(), CCInfo, Outs, /*IsRet=*/true,
                    nullptr, CC_Primate);

  if (CallConv == CallingConv::GHC && !PRLocs.empty())
    report_fatal_error("GHC functions return void only");

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = PRLocs.size(); i < e; ++i) {
    SDValue Val = OutVals[i];
    CCValAssign &VA = PRLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64) {
      // Handle returning f64 on PR32D with a soft float ABI.
      assert(VA.isRegLoc() && "Expected return via registers");
      SDValue SplitF64 = DAG.getNode(PrimateISD::SplitF64, DL,
                                     DAG.getVTList(MVT::i32, MVT::i32), Val);
      SDValue Lo = SplitF64.getValue(0);
      SDValue Hi = SplitF64.getValue(1);
      Register RegLo = VA.getLocReg();
      assert(RegLo < Primate::X31 && "Invalid register pair");
      Register RegHi = RegLo + 1;

      if (STI.isRegisterReservedByUser(RegLo) ||
          STI.isRegisterReservedByUser(RegHi))
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
            MF.getFunction(),
            "Return value register required, but has been reserved."});

      Chain = DAG.getCopyToReg(Chain, DL, RegLo, Lo, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegLo, MVT::i32));
      Chain = DAG.getCopyToReg(Chain, DL, RegHi, Hi, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegHi, MVT::i32));
    } else {
      // Handle a 'normal' return.
      Val = convertValVTToLocVT(DAG, Val, VA, DL, Subtarget);
      Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Glue);

      if (STI.isRegisterReservedByUser(VA.getLocReg()))
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
            MF.getFunction(),
            "Return value register required, but has been reserved."});

      // Guarantee that all emitted copies are stuck together.
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
    }
  }

  RetOps[0] = Chain; // Update chain.

  // Add the glue node if we have it.
  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }

  unsigned RetOpc = PrimateISD::RET_FLAG;
  // Interrupt service routines use different return instructions.
  const Function &Func = DAG.getMachineFunction().getFunction();
  if (Func.hasFnAttribute("interrupt")) {
    if (!Func.getReturnType()->isVoidTy())
      report_fatal_error(
          "Functions with the interrupt attribute must have void return type!");

    MachineFunction &MF = DAG.getMachineFunction();
    StringRef Kind =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

    if (Kind == "user")
      RetOpc = PrimateISD::URET_FLAG;
    else if (Kind == "supervisor")
      RetOpc = PrimateISD::SRET_FLAG;
    else
      RetOpc = PrimateISD::MRET_FLAG;
  }

  return DAG.getNode(RetOpc, DL, MVT::Other, RetOps);
}

void PrimateTargetLowering::validateCCReservedRegs(
    const SmallVectorImpl<std::pair<llvm::Register, llvm::SDValue>> &Regs,
    MachineFunction &MF) const {
  const Function &F = MF.getFunction();
  const PrimateSubtarget &STI = MF.getSubtarget<PrimateSubtarget>();

  if (llvm::any_of(Regs, [&STI](auto Reg) {
        return STI.isRegisterReservedByUser(Reg.first);
      }))
    F.getContext().diagnose(DiagnosticInfoUnsupported{
        F, "Argument register required, but has been reserved."});
}

bool PrimateTargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  return CI->isTailCall();
}

const char *PrimateTargetLowering::getTargetNodeName(unsigned Opcode) const {
#define NODE_NAME_CASE(NODE)                                                   \
  case PrimateISD::NODE:                                                         \
    return "PrimateISD::" #NODE;
  // clang-format off
  switch ((PrimateISD::NodeType)Opcode) {
  case PrimateISD::FIRST_NUMBER:
    break;
  NODE_NAME_CASE(RET_FLAG)
  NODE_NAME_CASE(URET_FLAG)
  NODE_NAME_CASE(SRET_FLAG)
  NODE_NAME_CASE(MRET_FLAG)
  NODE_NAME_CASE(CALL)
  NODE_NAME_CASE(SELECT_CC)
  NODE_NAME_CASE(BR_CC)
  NODE_NAME_CASE(BuildPairF64)
  NODE_NAME_CASE(SplitF64)
  NODE_NAME_CASE(TAIL)
  NODE_NAME_CASE(MULHSU)
  NODE_NAME_CASE(SLLW)
  NODE_NAME_CASE(SRAW)
  NODE_NAME_CASE(SRLW)
  NODE_NAME_CASE(DIVW)
  NODE_NAME_CASE(DIVUW)
  NODE_NAME_CASE(REMUW)
  NODE_NAME_CASE(ROLW)
  NODE_NAME_CASE(RORW)
  NODE_NAME_CASE(CLZW)
  NODE_NAME_CASE(CTZW)
  NODE_NAME_CASE(FSLW)
  NODE_NAME_CASE(FSRW)
  NODE_NAME_CASE(FSL)
  NODE_NAME_CASE(FSR)
  NODE_NAME_CASE(FMV_H_X)
  NODE_NAME_CASE(FMV_X_ANYEXTH)
  NODE_NAME_CASE(FMV_W_X_PR64)
  NODE_NAME_CASE(FMV_X_ANYEXTW_PR64)
  NODE_NAME_CASE(FCVT_W_PR64)
  NODE_NAME_CASE(FCVT_WU_PR64)
  NODE_NAME_CASE(READ_CYCLE_WIDE)
  NODE_NAME_CASE(GREV)
  NODE_NAME_CASE(GREVW)
  NODE_NAME_CASE(GORC)
  NODE_NAME_CASE(GORCW)
  NODE_NAME_CASE(SHFL)
  NODE_NAME_CASE(SHFLW)
  NODE_NAME_CASE(UNSHFL)
  NODE_NAME_CASE(UNSHFLW)
  NODE_NAME_CASE(BCOMPRESS)
  NODE_NAME_CASE(BCOMPRESSW)
  NODE_NAME_CASE(BDECOMPRESS)
  NODE_NAME_CASE(BDECOMPRESSW)
  // NODE_NAME_CASE(VMV_V_X_VL)
  // NODE_NAME_CASE(VFMV_V_F_VL)
  // NODE_NAME_CASE(VMV_X_S)
  // NODE_NAME_CASE(VMV_S_X_VL)
  // NODE_NAME_CASE(VFMV_S_F_VL)
  // NODE_NAME_CASE(SPLAT_VECTOR_I64)
  // NODE_NAME_CASE(SPLAT_VECTOR_SPLIT_I64_VL)
  // NODE_NAME_CASE(READ_VLENB)
  // NODE_NAME_CASE(TRUNCATE_VECTOR_VL)
  // NODE_NAME_CASE(VSLIDEUP_VL)
  // NODE_NAME_CASE(VSLIDE1UP_VL)
  // NODE_NAME_CASE(VSLIDEDOWN_VL)
  // NODE_NAME_CASE(VSLIDE1DOWN_VL)
  // NODE_NAME_CASE(VID_VL)
  // NODE_NAME_CASE(VFNCVT_ROD_VL)
  // NODE_NAME_CASE(VECREDUCE_ADD_VL)
  // NODE_NAME_CASE(VECREDUCE_UMAX_VL)
  // NODE_NAME_CASE(VECREDUCE_SMAX_VL)
  // NODE_NAME_CASE(VECREDUCE_UMIN_VL)
  // NODE_NAME_CASE(VECREDUCE_SMIN_VL)
  // NODE_NAME_CASE(VECREDUCE_AND_VL)
  // NODE_NAME_CASE(VECREDUCE_OR_VL)
  // NODE_NAME_CASE(VECREDUCE_XOR_VL)
  // NODE_NAME_CASE(VECREDUCE_FADD_VL)
  // NODE_NAME_CASE(VECREDUCE_SEQ_FADD_VL)
  // NODE_NAME_CASE(VECREDUCE_FMIN_VL)
  // NODE_NAME_CASE(VECREDUCE_FMAX_VL)
  // NODE_NAME_CASE(ADD_VL)
  // NODE_NAME_CASE(AND_VL)
  // NODE_NAME_CASE(MUL_VL)
  // NODE_NAME_CASE(OR_VL)
  // NODE_NAME_CASE(SDIV_VL)
  // NODE_NAME_CASE(SHL_VL)
  // NODE_NAME_CASE(SREM_VL)
  // NODE_NAME_CASE(SRA_VL)
  // NODE_NAME_CASE(SRL_VL)
  // NODE_NAME_CASE(SUB_VL)
  // NODE_NAME_CASE(UDIV_VL)
  // NODE_NAME_CASE(UREM_VL)
  // NODE_NAME_CASE(XOR_VL)
  // NODE_NAME_CASE(SADDSAT_VL)
  // NODE_NAME_CASE(UADDSAT_VL)
  // NODE_NAME_CASE(SSUBSAT_VL)
  // NODE_NAME_CASE(USUBSAT_VL)
  // NODE_NAME_CASE(FADD_VL)
  // NODE_NAME_CASE(FSUB_VL)
  // NODE_NAME_CASE(FMUL_VL)
  // NODE_NAME_CASE(FDIV_VL)
  // NODE_NAME_CASE(FNEG_VL)
  // NODE_NAME_CASE(FABS_VL)
  // NODE_NAME_CASE(FSQRT_VL)
  // NODE_NAME_CASE(FMA_VL)
  // NODE_NAME_CASE(FCOPYSIGN_VL)
  // NODE_NAME_CASE(SMIN_VL)
  // NODE_NAME_CASE(SMAX_VL)
  // NODE_NAME_CASE(UMIN_VL)
  // NODE_NAME_CASE(UMAX_VL)
  // NODE_NAME_CASE(FMINNUM_VL)
  // NODE_NAME_CASE(FMAXNUM_VL)
  // NODE_NAME_CASE(MULHS_VL)
  // NODE_NAME_CASE(MULHU_VL)
  // NODE_NAME_CASE(FP_TO_SINT_VL)
  // NODE_NAME_CASE(FP_TO_UINT_VL)
  // NODE_NAME_CASE(SINT_TO_FP_VL)
  // NODE_NAME_CASE(UINT_TO_FP_VL)
  // NODE_NAME_CASE(FP_EXTEND_VL)
  // NODE_NAME_CASE(FP_ROUND_VL)
  // NODE_NAME_CASE(VWMUL_VL)
  // NODE_NAME_CASE(VWMULU_VL)
  // NODE_NAME_CASE(SETCC_VL)
  // NODE_NAME_CASE(VSELECT_VL)
  // NODE_NAME_CASE(VMAND_VL)
  // NODE_NAME_CASE(VMOR_VL)
  // NODE_NAME_CASE(VMXOR_VL)
  // NODE_NAME_CASE(VMCLR_VL)
  // NODE_NAME_CASE(VMSET_VL)
  // NODE_NAME_CASE(VRGATHER_VX_VL)
  // NODE_NAME_CASE(VRGATHER_VV_VL)
  // NODE_NAME_CASE(VRGATHEREI16_VV_VL)
  // NODE_NAME_CASE(VSEXT_VL)
  // NODE_NAME_CASE(VZEXT_VL)
  // NODE_NAME_CASE(VPOPC_VL)
  NODE_NAME_CASE(VLE_VL)
  NODE_NAME_CASE(VSE_VL)
  NODE_NAME_CASE(READ_CSR)
  NODE_NAME_CASE(WRITE_CSR)
  NODE_NAME_CASE(SWAP_CSR)
  NODE_NAME_CASE(EXTRACT)
  NODE_NAME_CASE(INSERT)
  }
  // clang-format on
  return nullptr;
#undef NODE_NAME_CASE
}

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
PrimateTargetLowering::ConstraintType
PrimateTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'f':
    case 'v':
      return C_RegisterClass;
    case 'I':
    case 'J':
    case 'K':
      return C_Immediate;
    case 'A':
      return C_Memory;
    case 'S': // A symbolic address
      return C_Other;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
PrimateTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                  StringRef Constraint,
                                                  MVT VT) const {
  // First, see if this is a constraint that directly corresponds to a
  // Primate register class.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, &Primate::GPRRegClass);
    case 'f':
      if (Subtarget.hasStdExtZfh() && VT == MVT::f16)
        return std::make_pair(0U, &Primate::FPR16RegClass);
      if (Subtarget.hasStdExtF() && VT == MVT::f32)
        return std::make_pair(0U, &Primate::FPR32RegClass);
      if (Subtarget.hasStdExtD() && VT == MVT::f64)
        return std::make_pair(0U, &Primate::FPR64RegClass);
      break;
    case 'v':
      for (const auto *RC :
           {&Primate::VMRegClass, &Primate::VRRegClass, &Primate::VRM2RegClass,
            &Primate::VRM4RegClass, &Primate::VRM8RegClass}) {
        if (TRI->isTypeLegalForClass(*RC, VT.SimpleTy))
          return std::make_pair(0U, RC);
      }
      break;
    default:
      break;
    }
  }

  // Clang will correctly decode the usage of register name aliases into their
  // official names. However, other frontends like `rustc` do not. This allows
  // users of these frontends to use the ABI names for registers in LLVM-style
  // register constraints.
  unsigned XRegFromAlias = StringSwitch<unsigned>(Constraint.lower())
                               .Case("{zero}", Primate::X0)
                               .Case("{ra}", Primate::X1)
                               .Case("{sp}", Primate::X2)
                               .Case("{gp}", Primate::X3)
                               .Case("{tp}", Primate::X4)
                               .Case("{t0}", Primate::X5)
                               .Case("{t1}", Primate::X6)
                               .Case("{t2}", Primate::X7)
                               .Cases("{s0}", "{fp}", Primate::X8)
                               .Case("{s1}", Primate::X9)
                               .Case("{a0}", Primate::X10)
                               .Case("{a1}", Primate::X11)
                               .Case("{a2}", Primate::X12)
                               .Case("{a3}", Primate::X13)
                               .Case("{a4}", Primate::X14)
                               .Case("{a5}", Primate::X15)
                               .Case("{a6}", Primate::X16)
                               .Case("{a7}", Primate::X17)
                               .Case("{s2}", Primate::X18)
                               .Case("{s3}", Primate::X19)
                               .Case("{s4}", Primate::X20)
                               .Case("{s5}", Primate::X21)
                               .Case("{s6}", Primate::X22)
                               .Case("{s7}", Primate::X23)
                               .Case("{s8}", Primate::X24)
                               .Case("{s9}", Primate::X25)
                               .Case("{s10}", Primate::X26)
                               .Case("{s11}", Primate::X27)
                               .Case("{t3}", Primate::X28)
                               .Case("{t4}", Primate::X29)
                               .Case("{t5}", Primate::X30)
                               .Case("{t6}", Primate::X31)
                               .Default(Primate::NoRegister);
  if (XRegFromAlias != Primate::NoRegister)
    return std::make_pair(XRegFromAlias, &Primate::GPRRegClass);

  // Since TargetLowering::getRegForInlineAsmConstraint uses the name of the
  // TableGen record rather than the AsmName to choose registers for InlineAsm
  // constraints, plus we want to match those names to the widest floating point
  // register type available, manually select floating point registers here.
  //
  // The second case is the ABI name of the register, so that frontends can also
  // use the ABI names in register constraint lists.
  if (Subtarget.hasStdExtF()) {
    unsigned FReg = StringSwitch<unsigned>(Constraint.lower())
                        .Cases("{f0}", "{ft0}", Primate::F0_F)
                        .Cases("{f1}", "{ft1}", Primate::F1_F)
                        .Cases("{f2}", "{ft2}", Primate::F2_F)
                        .Cases("{f3}", "{ft3}", Primate::F3_F)
                        .Cases("{f4}", "{ft4}", Primate::F4_F)
                        .Cases("{f5}", "{ft5}", Primate::F5_F)
                        .Cases("{f6}", "{ft6}", Primate::F6_F)
                        .Cases("{f7}", "{ft7}", Primate::F7_F)
                        .Cases("{f8}", "{fs0}", Primate::F8_F)
                        .Cases("{f9}", "{fs1}", Primate::F9_F)
                        .Cases("{f10}", "{fa0}", Primate::F10_F)
                        .Cases("{f11}", "{fa1}", Primate::F11_F)
                        .Cases("{f12}", "{fa2}", Primate::F12_F)
                        .Cases("{f13}", "{fa3}", Primate::F13_F)
                        .Cases("{f14}", "{fa4}", Primate::F14_F)
                        .Cases("{f15}", "{fa5}", Primate::F15_F)
                        .Cases("{f16}", "{fa6}", Primate::F16_F)
                        .Cases("{f17}", "{fa7}", Primate::F17_F)
                        .Cases("{f18}", "{fs2}", Primate::F18_F)
                        .Cases("{f19}", "{fs3}", Primate::F19_F)
                        .Cases("{f20}", "{fs4}", Primate::F20_F)
                        .Cases("{f21}", "{fs5}", Primate::F21_F)
                        .Cases("{f22}", "{fs6}", Primate::F22_F)
                        .Cases("{f23}", "{fs7}", Primate::F23_F)
                        .Cases("{f24}", "{fs8}", Primate::F24_F)
                        .Cases("{f25}", "{fs9}", Primate::F25_F)
                        .Cases("{f26}", "{fs10}", Primate::F26_F)
                        .Cases("{f27}", "{fs11}", Primate::F27_F)
                        .Cases("{f28}", "{ft8}", Primate::F28_F)
                        .Cases("{f29}", "{ft9}", Primate::F29_F)
                        .Cases("{f30}", "{ft10}", Primate::F30_F)
                        .Cases("{f31}", "{ft11}", Primate::F31_F)
                        .Default(Primate::NoRegister);
    if (FReg != Primate::NoRegister) {
      assert(Primate::F0_F <= FReg && FReg <= Primate::F31_F && "Unknown fp-reg");
      if (Subtarget.hasStdExtD()) {
        unsigned RegNo = FReg - Primate::F0_F;
        unsigned DReg = Primate::F0_D + RegNo;
        return std::make_pair(DReg, &Primate::FPR64RegClass);
      }
      return std::make_pair(FReg, &Primate::FPR32RegClass);
    }
  }

  if (Subtarget.hasStdExtV()) {
    Register VReg = StringSwitch<Register>(Constraint.lower())
                        .Case("{v0}", Primate::V0)
                        .Case("{v1}", Primate::V1)
                        .Case("{v2}", Primate::V2)
                        .Case("{v3}", Primate::V3)
                        .Case("{v4}", Primate::V4)
                        .Case("{v5}", Primate::V5)
                        .Case("{v6}", Primate::V6)
                        .Case("{v7}", Primate::V7)
                        .Case("{v8}", Primate::V8)
                        .Case("{v9}", Primate::V9)
                        .Case("{v10}", Primate::V10)
                        .Case("{v11}", Primate::V11)
                        .Case("{v12}", Primate::V12)
                        .Case("{v13}", Primate::V13)
                        .Case("{v14}", Primate::V14)
                        .Case("{v15}", Primate::V15)
                        .Case("{v16}", Primate::V16)
                        .Case("{v17}", Primate::V17)
                        .Case("{v18}", Primate::V18)
                        .Case("{v19}", Primate::V19)
                        .Case("{v20}", Primate::V20)
                        .Case("{v21}", Primate::V21)
                        .Case("{v22}", Primate::V22)
                        .Case("{v23}", Primate::V23)
                        .Case("{v24}", Primate::V24)
                        .Case("{v25}", Primate::V25)
                        .Case("{v26}", Primate::V26)
                        .Case("{v27}", Primate::V27)
                        .Case("{v28}", Primate::V28)
                        .Case("{v29}", Primate::V29)
                        .Case("{v30}", Primate::V30)
                        .Case("{v31}", Primate::V31)
                        .Default(Primate::NoRegister);
    if (VReg != Primate::NoRegister) {
      if (TRI->isTypeLegalForClass(Primate::VMRegClass, VT.SimpleTy))
        return std::make_pair(VReg, &Primate::VMRegClass);
      if (TRI->isTypeLegalForClass(Primate::VRRegClass, VT.SimpleTy))
        return std::make_pair(VReg, &Primate::VRRegClass);
      for (const auto *RC :
           {&Primate::VRM2RegClass, &Primate::VRM4RegClass, &Primate::VRM8RegClass}) {
        if (TRI->isTypeLegalForClass(*RC, VT.SimpleTy)) {
          VReg = TRI->getMatchingSuperReg(VReg, Primate::sub_vrm1_0, RC);
          return std::make_pair(VReg, RC);
        }
      }
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

InlineAsm::ConstraintCode
PrimateTargetLowering::getInlineAsmMemConstraint(StringRef ConstraintCode) const {
  // Currently only support length 1 constraints.
  if (ConstraintCode.size() == 1) {
    switch (ConstraintCode[0]) {
    case 'A':
      return InlineAsm::ConstraintCode::A;
    default:
      break;
    }
  }

  return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
}

void PrimateTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, StringRef Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  // Currently only support length 1 constraints.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'I':
      // Validate & create a 12-bit signed immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getSExtValue();
        if (isInt<12>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    case 'J':
      // Validate & create an integer zero operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (C->getZExtValue() == 0)
          Ops.push_back(
              DAG.getTargetConstant(0, SDLoc(Op), Subtarget.getXLenVT()));
      return;
    case 'K':
      // Validate & create a 5-bit unsigned immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getZExtValue();
        if (isUInt<5>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    case 'S':
      if (const auto *GA = dyn_cast<GlobalAddressSDNode>(Op)) {
        Ops.push_back(DAG.getTargetGlobalAddress(GA->getGlobal(), SDLoc(Op),
                                                 GA->getValueType(0)));
      } else if (const auto *BA = dyn_cast<BlockAddressSDNode>(Op)) {
        Ops.push_back(DAG.getTargetBlockAddress(BA->getBlockAddress(),
                                                BA->getValueType(0)));
      }
      return;
    default:
      break;
    }
  }
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

Instruction *PrimateTargetLowering::emitLeadingFence(IRBuilderBase &Builder,
                                                   Instruction *Inst,
                                                   AtomicOrdering Ord) const {
  if (isa<LoadInst>(Inst) && Ord == AtomicOrdering::SequentiallyConsistent)
    return Builder.CreateFence(Ord);
  if (isa<StoreInst>(Inst) && isReleaseOrStronger(Ord))
    return Builder.CreateFence(AtomicOrdering::Release);
  return nullptr;
}

Instruction *PrimateTargetLowering::emitTrailingFence(IRBuilderBase &Builder,
                                                    Instruction *Inst,
                                                    AtomicOrdering Ord) const {
  if (isa<LoadInst>(Inst) && isAcquireOrStronger(Ord))
    return Builder.CreateFence(AtomicOrdering::Acquire);
  return nullptr;
}

TargetLowering::AtomicExpansionKind
PrimateTargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const {
  // atomicrmw {fadd,fsub} must be expanded to use compare-exchange, as floating
  // point operations can't be used in an lr/sc sequence without breaking the
  // forward-progress guarantee.
  if (AI->isFloatingPointOperation())
    return AtomicExpansionKind::CmpXChg;

  unsigned Size = AI->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

static Intrinsic::ID
getIntrinsicForMaskedAtomicRMWBinOp(unsigned XLen, AtomicRMWInst::BinOp BinOp) {
  if (XLen == 32) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::primate_masked_atomicrmw_xchg_i32;
    case AtomicRMWInst::Add:
      return Intrinsic::primate_masked_atomicrmw_add_i32;
    case AtomicRMWInst::Sub:
      return Intrinsic::primate_masked_atomicrmw_sub_i32;
    case AtomicRMWInst::Nand:
      return Intrinsic::primate_masked_atomicrmw_nand_i32;
    case AtomicRMWInst::Max:
      return Intrinsic::primate_masked_atomicrmw_max_i32;
    case AtomicRMWInst::Min:
      return Intrinsic::primate_masked_atomicrmw_min_i32;
    case AtomicRMWInst::UMax:
      return Intrinsic::primate_masked_atomicrmw_umax_i32;
    case AtomicRMWInst::UMin:
      return Intrinsic::primate_masked_atomicrmw_umin_i32;
    }
  }

  if (XLen == 64) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::primate_masked_atomicrmw_xchg_i64;
    case AtomicRMWInst::Add:
      return Intrinsic::primate_masked_atomicrmw_add_i64;
    case AtomicRMWInst::Sub:
      return Intrinsic::primate_masked_atomicrmw_sub_i64;
    case AtomicRMWInst::Nand:
      return Intrinsic::primate_masked_atomicrmw_nand_i64;
    case AtomicRMWInst::Max:
      return Intrinsic::primate_masked_atomicrmw_max_i64;
    case AtomicRMWInst::Min:
      return Intrinsic::primate_masked_atomicrmw_min_i64;
    case AtomicRMWInst::UMax:
      return Intrinsic::primate_masked_atomicrmw_umax_i64;
    case AtomicRMWInst::UMin:
      return Intrinsic::primate_masked_atomicrmw_umin_i64;
    }
  }

  llvm_unreachable("Unexpected XLen\n");
}

Value *PrimateTargetLowering::emitMaskedAtomicRMWIntrinsic(
    IRBuilderBase &Builder, AtomicRMWInst *AI, Value *AlignedAddr, Value *Incr,
    Value *Mask, Value *ShiftAmt, AtomicOrdering Ord) const {
  unsigned XLen = Subtarget.getXLen();
  Value *Ordering =
      Builder.getIntN(XLen, static_cast<uint64_t>(AI->getOrdering()));
  Type *Tys[] = {AlignedAddr->getType()};
  Function *LrwOpScwLoop = Intrinsic::getDeclaration(
      AI->getModule(),
      getIntrinsicForMaskedAtomicRMWBinOp(XLen, AI->getOperation()), Tys);

  if (XLen == 64) {
    Incr = Builder.CreateSExt(Incr, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    ShiftAmt = Builder.CreateSExt(ShiftAmt, Builder.getInt64Ty());
  }

  Value *Result;

  // Must pass the shift amount needed to sign extend the loaded value prior
  // to performing a signed comparison for min/max. ShiftAmt is the number of
  // bits to shift the value into position. Pass XLen-ShiftAmt-ValWidth, which
  // is the number of bits to left+right shift the value in order to
  // sign-extend.
  if (AI->getOperation() == AtomicRMWInst::Min ||
      AI->getOperation() == AtomicRMWInst::Max) {
    const DataLayout &DL = AI->getModule()->getDataLayout();
    unsigned ValWidth =
        DL.getTypeStoreSizeInBits(AI->getValOperand()->getType());
    Value *SextShamt =
        Builder.CreateSub(Builder.getIntN(XLen, XLen - ValWidth), ShiftAmt);
    Result = Builder.CreateCall(LrwOpScwLoop,
                                {AlignedAddr, Incr, Mask, SextShamt, Ordering});
  } else {
    Result =
        Builder.CreateCall(LrwOpScwLoop, {AlignedAddr, Incr, Mask, Ordering});
  }

  if (XLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

TargetLowering::AtomicExpansionKind
PrimateTargetLowering::shouldExpandAtomicCmpXchgInIR(
    AtomicCmpXchgInst *CI) const {
  unsigned Size = CI->getCompareOperand()->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

Value *PrimateTargetLowering::emitMaskedAtomicCmpXchgIntrinsic(
    IRBuilderBase &Builder, AtomicCmpXchgInst *CI, Value *AlignedAddr,
    Value *CmpVal, Value *NewVal, Value *Mask, AtomicOrdering Ord) const {
  unsigned XLen = Subtarget.getXLen();
  Value *Ordering = Builder.getIntN(XLen, static_cast<uint64_t>(Ord));
  Intrinsic::ID CmpXchgIntrID = Intrinsic::primate_masked_cmpxchg_i32;
  if (XLen == 64) {
    CmpVal = Builder.CreateSExt(CmpVal, Builder.getInt64Ty());
    NewVal = Builder.CreateSExt(NewVal, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    CmpXchgIntrID = Intrinsic::primate_masked_cmpxchg_i64;
  }
  Type *Tys[] = {AlignedAddr->getType()};
  Function *MaskedCmpXchg =
      Intrinsic::getDeclaration(CI->getModule(), CmpXchgIntrID, Tys);
  Value *Result = Builder.CreateCall(
      MaskedCmpXchg, {AlignedAddr, CmpVal, NewVal, Mask, Ordering});
  if (XLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

bool PrimateTargetLowering::shouldRemoveExtendFromGSIndex(SDValue Extend, EVT VT) const {
  return false;
}

bool PrimateTargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                     EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f16:
    return Subtarget.hasStdExtZfh();
  case MVT::f32:
    return Subtarget.hasStdExtF();
  case MVT::f64:
    return Subtarget.hasStdExtD();
  default:
    break;
  }

  return false;
}

Register PrimateTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return Primate::X10;
}

Register PrimateTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return Primate::X11;
}

bool PrimateTargetLowering::shouldExtendTypeInLibCall(EVT Type) const {
  // Return false to suppress the unnecessary extensions if the LibCall
  // arguments or return value is f32 type for LP64 ABI.
  PrimateABI::ABI ABI = Subtarget.getTargetABI();
  if (ABI == PrimateABI::ABI_LP64 && (Type == MVT::f32))
    return false;

  return true;
}

bool PrimateTargetLowering::shouldSignExtendTypeInLibCall(EVT Type, bool IsSigned) const {
  if (Subtarget.is64Bit() && Type == MVT::i32)
    return true;

  return IsSigned;
}

bool PrimateTargetLowering::decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                                 SDValue C) const {
  // Check integral scalar types.
  if (VT.isScalarInteger()) {
    // Omit the optimization if the sub target has the M extension and the data
    // size exceeds XLen.
    if (Subtarget.hasStdExtM() && VT.getSizeInBits() > Subtarget.getXLen())
      return false;
    if (auto *ConstNode = dyn_cast<ConstantSDNode>(C.getNode())) {
      // Break the MUL to a SLLI and an ADD/SUB.
      const APInt &Imm = ConstNode->getAPIntValue();
      if ((Imm + 1).isPowerOf2() || (Imm - 1).isPowerOf2() ||
          (1 - Imm).isPowerOf2() || (-1 - Imm).isPowerOf2())
        return true;
      // Omit the following optimization if the sub target has the M extension
      // and the data size >= XLen.
      if (Subtarget.hasStdExtM() && VT.getSizeInBits() >= Subtarget.getXLen())
        return false;
      // Break the MUL to two SLLI instructions and an ADD/SUB, if Imm needs
      // a pair of LUI/ADDI.
      if (!Imm.isSignedIntN(12) && Imm.countTrailingZeros() < 12) {
        APInt ImmS = Imm.ashr(Imm.countTrailingZeros());
        if ((ImmS + 1).isPowerOf2() || (ImmS - 1).isPowerOf2() ||
            (1 - ImmS).isPowerOf2())
        return true;
      }
    }
  }

  return false;
}

bool PrimateTargetLowering::allowsMisalignedMemoryAccesses(
    EVT VT, unsigned AddrSpace, Align Alignment, MachineMemOperand::Flags Flags,
    unsigned *Fast) const {
  if (!VT.isVector())
    return false;

  EVT ElemVT = VT.getVectorElementType();
  if (Alignment >= ElemVT.getStoreSize()) {
    if (Fast)
      *Fast = 1;
    return true;
  }

  return false;
}

bool PrimateTargetLowering::splitValueIntoRegisterParts(
    SelectionDAG &DAG, const SDLoc &DL, SDValue Val, SDValue *Parts,
    unsigned NumParts, MVT PartVT, std::optional<CallingConv::ID> CC) const {
  bool IsABIRegCopy = CC.has_value();
  EVT ValueVT = Val.getValueType();
  if (IsABIRegCopy && ValueVT == MVT::f16 && PartVT == MVT::f32) {
    // Cast the f16 to i16, extend to i32, pad with ones to make a float nan,
    // and cast to f32.
    Val = DAG.getNode(ISD::BITCAST, DL, MVT::i16, Val);
    Val = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Val);
    Val = DAG.getNode(ISD::OR, DL, MVT::i32, Val,
                      DAG.getConstant(0xFFFF0000, DL, MVT::i32));
    Val = DAG.getNode(ISD::BITCAST, DL, MVT::f32, Val);
    Parts[0] = Val;
    return true;
  }

  if (ValueVT.isScalableVector() && PartVT.isScalableVector()) {
    LLVMContext &Context = *DAG.getContext();
    EVT ValueEltVT = ValueVT.getVectorElementType();
    EVT PartEltVT = PartVT.getVectorElementType();
    unsigned ValueVTBitSize = ValueVT.getSizeInBits().getKnownMinValue();
    unsigned PartVTBitSize = PartVT.getSizeInBits().getKnownMinValue();
    if (PartVTBitSize % ValueVTBitSize == 0) {
      // If the element types are different, bitcast to the same element type of
      // PartVT first.
      if (ValueEltVT != PartEltVT) {
        unsigned Count = ValueVTBitSize / PartEltVT.getSizeInBits();
        assert(Count != 0 && "The number of element should not be zero.");
        EVT SameEltTypeVT =
            EVT::getVectorVT(Context, PartEltVT, Count, /*IsScalable=*/true);
        Val = DAG.getNode(ISD::BITCAST, DL, SameEltTypeVT, Val);
      }
      Val = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, PartVT, DAG.getUNDEF(PartVT),
                        Val, DAG.getConstant(0, DL, Subtarget.getXLenVT()));
      Parts[0] = Val;
      return true;
    }
  }
  return false;
}

SDValue PrimateTargetLowering::joinRegisterPartsIntoValue(
    SelectionDAG &DAG, const SDLoc &DL, const SDValue *Parts, unsigned NumParts,
    MVT PartVT, EVT ValueVT, std::optional<CallingConv::ID> CC) const {
  bool IsABIRegCopy = CC.has_value();
  if (IsABIRegCopy && ValueVT == MVT::f16 && PartVT == MVT::f32) {
    SDValue Val = Parts[0];

    // Cast the f32 to i32, truncate to i16, and cast back to f16.
    Val = DAG.getNode(ISD::BITCAST, DL, MVT::i32, Val);
    Val = DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, Val);
    Val = DAG.getNode(ISD::BITCAST, DL, MVT::f16, Val);
    return Val;
  }

  if (ValueVT.isScalableVector() && PartVT.isScalableVector()) {
    LLVMContext &Context = *DAG.getContext();
    SDValue Val = Parts[0];
    EVT ValueEltVT = ValueVT.getVectorElementType();
    EVT PartEltVT = PartVT.getVectorElementType();
    unsigned ValueVTBitSize = ValueVT.getSizeInBits().getKnownMinValue();
    unsigned PartVTBitSize = PartVT.getSizeInBits().getKnownMinValue();
    if (PartVTBitSize % ValueVTBitSize == 0) {
      EVT SameEltTypeVT = ValueVT;
      // If the element types are different, convert it to the same element type
      // of PartVT.
      if (ValueEltVT != PartEltVT) {
        unsigned Count = ValueVTBitSize / PartEltVT.getSizeInBits();
        assert(Count != 0 && "The number of element should not be zero.");
        SameEltTypeVT =
            EVT::getVectorVT(Context, PartEltVT, Count, /*IsScalable=*/true);
      }
      Val = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SameEltTypeVT, Val,
                        DAG.getConstant(0, DL, Subtarget.getXLenVT()));
      if (ValueEltVT != PartEltVT)
        Val = DAG.getNode(ISD::BITCAST, DL, ValueVT, Val);
      return Val;
    }
  }
  return SDValue();
}

#define GET_REGISTER_MATCHER
#include "PrimateGenAsmMatcher.inc"

Register
PrimateTargetLowering::getRegisterByName(const char *RegName, LLT VT,
                                       const MachineFunction &MF) const {
  Register Reg = MatchRegisterAltName(RegName);
  if (Reg == Primate::NoRegister)
    Reg = MatchRegisterName(RegName);
  if (Reg == Primate::NoRegister)
    report_fatal_error(
        Twine("Invalid register name \"" + StringRef(RegName) + "\"."));
  BitVector ReservedRegs = Subtarget.getRegisterInfo()->getReservedRegs(MF);
  if (!ReservedRegs.test(Reg) && !Subtarget.isRegisterReservedByUser(Reg))
    report_fatal_error(Twine("Trying to obtain non-reserved register \"" +
                             StringRef(RegName) + "\"."));
  return Reg;
}

namespace llvm {
namespace PrimateVIntrinsicsTable {

#define GET_PrimateVIntrinsicsTable_IMPL
#include "PrimateGenSearchableTables.inc"

} // namespace PrimateVIntrinsicsTable

} // namespace llvm
