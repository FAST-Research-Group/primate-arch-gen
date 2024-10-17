
  //===- PrimateInstrInfoBFU.td - Target Description for Primate *- tablegen -*-===//
  //
  // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
  // See https://llvm.org/LICENSE.txt for license information.
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //
  //===-------------------------------------------------------------------------===//
  //
  // This file describes the Primate BFU instructions in TableGen format.
  //
  //===-------------------------------------------------------------------------===//


  /*
Initial selection DAG: %bb.0 '_Z7mul_acc11MAC_input_t:entry'
SelectionDAG has 21 nodes:
  t0: ch,glue = EntryToken
  t3: i32 = Constant<0>

 |--t17: i32 = add nsw t16, t9
 |    
 |--|--t16: i32 = mul nsw t15, t12
 |  |   
 |  |--|--t15: i32,ch = load<(dereferenceable load (s32) from %ir.bb, !tbaa !15)> t0, t14, undef:i32
 |  |  |
 |  |  |----t14: i32 = add nuw t2, Constant:i32<12>
 |  |
 |  |--|--t12: i32,ch = load<(dereferenceable load (s32) from %ir.aa, !tbaa !14)> t0, t11, undef:i32
 |     |   
 |     |----t11: i32 = add nuw t2, Constant:i32<4>
 |   
 |-----t9: i32 = mul nsw t8, t5
 |      t8: i32,ch = load<(dereferenceable load (s32) from %ir.b, !tbaa !13)> t0, t7, undef:i32
 |        t7: i32 = add nuw t2, Constant:i32<8>
 |      t5: i32,ch = load<(dereferenceable load (s32) from %ir.vec_in, !tbaa !8)> t0, t2, undef:i32
 |        t2: i32,ch = CopyFromReg t0, Register:i32 %0
  
  t20: ch = PrimateISD::RET_FLAG t19, Register:i32 $x10, t19:1
    t19: ch,glue = CopyToReg t0, Register:i32 $x10, t17


  t15 will become load with imm = 12
  */

  // def BFU0 :
  //     PRInstI<0b000, OPC_PR_ASCII, (outs BaseAddr:$rd), (ins BaseAddr:$rs1),
  //         "bfu0", "$rd, $rs1">, Sched<[WriteIALU, ReadIALU]> {
  //           let IsBFUInstruction = 1;
  //           let imm12 = 0;
  //         }

let Itinerary = ItinBlue0 in
let hasSideEffects = 1, mayLoad = 1, mayStore = 0 in
def BFU0 : Pseudo<(outs GPR:$rd), (ins GPR:$rs1), [], 
                  "bfu0", "$rd, $rs1">{
  let IsBFUInstruction = 1;
  // let imm12 = 0;
}


  // def : Pat<(int_primate_BFU_0 BaseAddr:$rs1), (BFU0 BaseAddr:$rs1)>;
  
// def : Pat<
//   (i32 (add 
//     (i32 (mul
//       (i32 (load BaseAddr:$rs)),
//       (i32 (load (i32 (add BaseAddr:$rs, 8))))
//     )), 
//     (i32 (mul
//       (i32 (load (i32 (add BaseAddr:$rs, 4)))),
//       (i32 (load (i32 (add BaseAddr:$rs, 12))))
//     ))
//   )),
//   (BFU0 BaseAddr:$rs)>;
  
let AddedComplexity = 1000 in
def : Pat<
  (i32 (add (i32 (mul (i32 (load BaseAddr:$rs)),
                      (i32 (load (i32 (add BaseAddr:$rs, 4)))))),
            (i32 (load (i32 (add BaseAddr:$rs, 8)))))),
  (BFU0 BaseAddr:$rs)>;

