//===- PrimateTargetDefEmitter.cpp - Generate lists of Primate CPUs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits the include file needed by the target
// parser to parse the Primate CPUs.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PrimateISAInfo.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

using ISAInfoTy = llvm::Expected<std::unique_ptr<PrimateISAInfo>>;

// We can generate march string from target features as what has been described
// in Primate ISA specification 
// Naming Conventions'.
//
// This is almost the same as PrimateFeatures::parseFeatureBits, except that we
// get feature name from feature records instead of feature bits.
static std::string getMArch(const Record &Rec) {
  std::vector<std::string> FeatureVector;
  unsigned XLen = 32;

  // Convert features to FeatureVector.
  for (auto *Feature : Rec.getValueAsListOfDefs("Features")) {
    StringRef FeatureName = Feature->getValueAsString("Name");
    if (llvm::PrimateISAInfo::isSupportedExtensionFeature(FeatureName))
      FeatureVector.push_back((Twine("+") + FeatureName).str());
    else if (FeatureName == "64bit")
      XLen = 64;
  }

  ISAInfoTy ISAInfo = llvm::PrimateISAInfo::parseFeatures(XLen, FeatureVector);
  if (!ISAInfo)
    report_fatal_error("Invalid features");

  // PrimateISAInfo::toString will generate a march string with all the extensions
  // we have added to it.
  return (*ISAInfo)->toString();
}

static void EmitPrimateTargetDef(RecordKeeper &RK, raw_ostream &OS) {
  OS << "#ifndef PROC\n"
     << "#define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_UNALIGNED_ACCESS)\n"
     << "#endif\n\n";

  // Iterate on all definition records.
  for (const Record *Rec : RK.getAllDerivedDefinitions("PrimateProcessorModel")) {
    std::string MArch = Rec->getValueAsString("DefaultMarch").str();

    // Compute MArch from features if we don't specify it.
    if (MArch.empty())
      MArch = getMArch(*Rec);

    const bool FastUnalignedAccess =
        any_of(Rec->getValueAsListOfDefs("Features"), [&](auto &Feature) {
          return Feature->getValueAsString("Name") == "fast-unaligned-access";
        });

    OS << "PROC(" << Rec->getName() << ", "
       << "{\"" << Rec->getValueAsString("Name") << "\"}, "
       << "{\"" << MArch << "\"}, " << FastUnalignedAccess << ")\n";
  }
  OS << "\n#undef PROC\n";
  OS << "\n";
  OS << "#ifndef TUNE_PROC\n"
     << "#define TUNE_PROC(ENUM, NAME)\n"
     << "#endif\n\n";
  OS << "TUNE_PROC(GENERIC, \"generic\")\n";

  for (const Record *Rec :
       RK.getAllDerivedDefinitions("PrimateTuneProcessorModel")) {
    OS << "TUNE_PROC(" << Rec->getName() << ", "
       << "\"" << Rec->getValueAsString("Name") << "\")\n";
  }

  OS << "\n#undef TUNE_PROC\n";
}

static TableGen::Emitter::Opt X("gen-primate-target-def", EmitPrimateTargetDef,
                                "Generate the list of CPU for Primate");
