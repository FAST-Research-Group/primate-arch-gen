add_llvm_component_group(Primate)

set(LLVM_TARGET_DEFINITIONS Primate.td)

tablegen(LLVM PrimateGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM PrimateGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM PrimateGenCompressInstEmitter.inc -gen-compress-inst-emitter)
tablegen(LLVM PrimateGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM PrimateGenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM PrimateGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM PrimateGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM PrimateGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM PrimateGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM PrimateGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM PrimateGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM PrimateGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM PrimateGenSubtargetInfo.inc -gen-subtarget)

set(LLVM_TARGET_DEFINITIONS PrimateGISel.td)
tablegen(LLVM PrimateGenGlobalISel.inc -gen-global-isel)
# tablegen(LLVM PrimateGenO0PreLegalizeGICombiner.inc -gen-global-isel-combiner
#               -combiners="PrimateO0PreLegalizerCombiner")
# tablegen(LLVM PrimateGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
#               -combiners="PrimatePreLegalizerCombiner")
# tablegen(LLVM PrimateGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
#               -combiners="PrimatePostLegalizerCombiner")

add_public_tablegen_target(PrimateCommonTableGen)

add_llvm_target(PrimateCodeGen
  PrimateAsmPrinter.cpp
  PrimateCallLowering.cpp
  PrimateExpandAtomicPseudoInsts.cpp
  PrimateExpandPseudoInsts.cpp
  PrimateFrameLowering.cpp
  PrimateInstrInfo.cpp
  PrimateInstructionSelector.cpp
  PrimateISelDAGToDAG.cpp
  PrimateInsertBFUCall.cpp
  PrimateISelLowering.cpp
  PrimateLegalizerInfo.cpp
  PrimateMCInstLower.cpp
  PrimateMergeBaseOffset.cpp
  PrimateRegisterBankInfo.cpp
  PrimateRegisterInfo.cpp
  PrimateSubtarget.cpp
  PrimateTargetMachine.cpp
  PrimateTargetObjectFile.cpp
  PrimateTargetTransformInfo.cpp
  PrimateVLIWPacketizer.cpp
  PrimateStructToRegPass.cpp
  PrimateGEPFilter.cpp
  PrimateStructToAggre.cpp
  PrimateIntrinsicPromotion.cpp
  PrimateCustomSchedule.cpp
  PrimateRegisterNormalize.cpp
  PrimateScheduleStrategy.cpp
  PrimateExtMerge.cpp
  PrimateOpMerge.cpp
  PrimatePacketLegalizer.cpp
  PrimateBFUTypeFindingPass.cpp
  PrimateMachineFunctionInfo.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  Core
  CodeGen
  MC
  PrimateDesc
  PrimateInfo
  SelectionDAG
  Support
  Target
  GlobalISel

  ADD_TO_COMPONENT
  Primate
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
