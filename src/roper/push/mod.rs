use std::fmt;

use falcon::il;
use hashbrown::HashMap;
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub use creature::Creature;

use crate::configure::Config;
use crate::emulator::loader;
use crate::emulator::loader::get_static_memory_image;
use crate::emulator::register_pattern::RegisterPattern;
use crate::util::architecture::{read_integer, write_integer, Perms};

pub mod evaluation;

pub type Stack<T> = Vec<T>;

pub type Input = Type;
pub type Output = Type;

// TODO: define a distribution over these ops. WordConst should comprise about half
// of any genome, I think, since without that, there's no ROP chain.

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Buffer {
    Pointer(u64),
    Vector(Vec<u8>),
}

/// Virtual
/// Instruction
/// Network
/// Domain
/// Specific
/// Language
///
/// VINDSL
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    BoolConst(bool),
    BoolAnd,
    BoolOr,
    BoolNot,

    WordConst(u64),
    WordLess,
    WordEqual,
    WordAdd,
    WordSub,
    WordAnd,
    WordOr,
    WordXor,
    WordDiv,
    WordMul,
    WordMod,
    WordIncr,
    WordDecr,
    WordOnes,
    WordShl,
    WordShr,

    // Treating the Word as an address
    WordDeref,
    WordSearch,
    WordDisRegRead,
    WordDisRegWrite,
    WordDisInstId,
    //WordReadRegs,
    //WordWriteRegs,
    // TODO: there should be a way to map addresses to IL CFG Nodes
    WordToFloat,

    WordReadable,
    WordWriteable,
    WordExecutable,

    WordToGadget,
    GadgetToWord,

    BufConst(Buffer),
    BufAt,
    // constant, by address
    BufLen,
    BufIndex,
    BufPack,
    BufHead,
    BufTail,
    //BufSearch,
    BufDisRegRead,
    BufDisRegWrite,
    BufDisInstId,

    // floats need to be encoded as u64 to preserve Hash
    FloatConst(u64),
    FloatLog,
    FloatSin,
    FloatCos,
    FloatTan,
    FloatTanh,
    FloatToWord,
    FloatLess,

    CodeQuote,
    CodeDoAll,

    // Falcon Instructions
    InstIsBranch,
    InstIsStore,
    InstIsLoad,
    InstAddr,

    // BlockConst(&'static il::Block),
    BlockInsts,
    BlockAddr,
    BlockSuccs,
    BlockPreds,

    //FuncConst(&'static il::Function),
    FuncNamed(String),
    FuncBlocks,
    // FuncConstants,
    // FuncEntry,
    // FuncExit,
    FuncAddr,
    ExprEval,

    ExecIf,
    // combinators
    ExecK,
    ExecS,
    ExecY,

    Nop,

    Eq(Type),
    Rot(Type),
    Swap(Type),
    Drop(Type),
    Dup(Type),

    List(Vec<Op>),
}

static NON_CONSTANT_OPS: [Op; 109] = [
    Op::BoolAnd,
    Op::BoolOr,
    Op::BoolNot,
    Op::WordLess,
    Op::WordEqual,
    Op::WordAdd,
    Op::WordSub,
    Op::WordAnd,
    Op::WordOr,
    Op::WordXor,
    Op::WordDiv,
    Op::WordMul,
    Op::WordMod,
    Op::WordOnes,
    Op::WordIncr,
    Op::WordDecr,
    Op::WordShl,
    Op::WordShr,
    Op::WordDeref,
    Op::WordSearch,
    // Op::WordReadRegs,
    // Op::WordWriteRegs,
    // Op::WordToFloat,
    Op::WordReadable,
    Op::WordWriteable,
    Op::WordExecutable,
    Op::WordDisRegRead,
    Op::WordDisRegWrite,
    Op::WordDisInstId,
    Op::WordToGadget,
    Op::GadgetToWord,
    //Op::BufConst(&'static [u8]),
    // Op::BufAt,
    // Op::BufLen,
    // Op::BufIndex,
    // Op::BufPack,
    // Op::BufHead,
    // Op::BufTail,
    // Op::BufSearch,
    // floats need to be encoded as u64 to preserve Op::Hash
    // Op::FloatLog,
    // Op::FloatSin,
    // Op::FloatCos,
    // Op::FloatTan,
    // Op::FloatTanh,
    // Op::FloatToWord,
    // Op::FloatLess,
    Op::CodeQuote,
    Op::CodeDoAll,
    Op::InstIsBranch,
    Op::InstIsStore,
    Op::InstIsLoad,
    Op::InstAddr,
    Op::BlockInsts,
    Op::BlockAddr,
    Op::BlockSuccs,
    Op::BlockPreds,
    Op::FuncBlocks,
    // Op::FuncConstants,
    // Op::FuncEntry,
    // Op::FuncExit,
    // Op::FuncAddr,
    Op::ExprEval,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecIf,
    Op::ExecK,
    Op::ExecS,
    Op::ExecY,
    Op::Nop,
    // Generics
    Op::Eq(Type::Word),
    Op::Rot(Type::Word),
    Op::Swap(Type::Word),
    Op::Drop(Type::Word),
    Op::Dup(Type::Word),
    Op::Eq(Type::Bool),
    Op::Rot(Type::Bool),
    Op::Swap(Type::Bool),
    Op::Drop(Type::Bool),
    Op::Dup(Type::Bool),
    Op::Eq(Type::Buf),
    Op::Rot(Type::Buf),
    Op::Swap(Type::Buf),
    Op::Drop(Type::Buf),
    Op::Dup(Type::Buf),
    Op::Eq(Type::Scalar),
    Op::Rot(Type::Scalar),
    Op::Swap(Type::Scalar),
    Op::Drop(Type::Scalar),
    Op::Dup(Type::Scalar),
    // Op::Eq(Type::Float),
    // Op::Rot(Type::Float),
    // Op::Swap(Type::Float),
    // Op::Drop(Type::Float),
    // Op::Dup(Type::Float),
    Op::Eq(Type::Expression),
    Op::Rot(Type::Expression),
    Op::Swap(Type::Expression),
    Op::Drop(Type::Expression),
    Op::Dup(Type::Expression),
    Op::Eq(Type::Block),
    Op::Rot(Type::Block),
    Op::Swap(Type::Block),
    Op::Drop(Type::Block),
    Op::Dup(Type::Block),
    Op::Eq(Type::Instruction),
    Op::Rot(Type::Instruction),
    Op::Swap(Type::Instruction),
    Op::Drop(Type::Instruction),
    Op::Dup(Type::Instruction),
    Op::Eq(Type::Function),
    Op::Rot(Type::Function),
    Op::Swap(Type::Function),
    Op::Drop(Type::Function),
    Op::Dup(Type::Function),
    Op::Eq(Type::Gadget),
    Op::Rot(Type::Gadget),
    Op::Swap(Type::Gadget),
    Op::Drop(Type::Gadget),
    Op::Dup(Type::Gadget),
    Op::Eq(Type::Exec),
    Op::Rot(Type::Exec),
    Op::Swap(Type::Exec),
    Op::Drop(Type::Exec),
    // Op::Dup(Type::Exec),
    Op::Eq(Type::Code),
    Op::Rot(Type::Code),
    Op::Swap(Type::Code),
    Op::Drop(Type::Code),
    Op::Dup(Type::Code),
];

fn random_ops<R: Rng>(rng: &mut R, config: &Config) -> Vec<Op> {
    let mut ops = Vec::new();

    let count = rng.gen_range(config.push_vm.min_len, config.push_vm.max_len);

    let memory = get_static_memory_image();
    let program = memory.il_program.as_ref().unwrap();
    let function_names: Vec<String> = program
        .functions()
        .into_iter()
        .map(il::Function::name)
        .collect();

    for _ in 0..count {
        if rng.gen_range(0.0, 1.0) < config.push_vm.literal_rate {
            match rng.gen_range(0, 2) {
                0 => {
                    // functions
                    let f = function_names
                        .choose(rng)
                        .expect("Failed to choose a random function")
                        .clone();
                    ops.push(Op::FuncNamed(f))
                }
                1 => {
                    // addresses
                    let addr = config
                        .roper
                        .soup
                        .as_ref()
                        .expect("No soup?!")
                        .choose(rng)
                        .expect("Failed to choose word from soup.");
                    ops.push(Op::WordConst(*addr))
                }
                // 3 => {
                //     // float
                //     ops.push(Op::FloatConst(rng.gen::<f64>().to_bits()))
                // }
                _ => unreachable!("nope"),
            };
        } else {
            let op = NON_CONSTANT_OPS
                .choose(rng)
                .expect("Failed to choose random op")
                .clone();
            ops.push(op)
        }
    }

    ops
}

// TODO: add data and control flow graph operations, using Falcon.
impl Op {
    pub fn eval(&self, mach: &mut MachineState) {
        use Op::*;
        use Val::*;
        match self {
            Nop => {}
            // Code manipulation
            List(ref ops) => {
                for op in ops {
                    mach.push(Exec(op.clone()))
                }
            }
            // Combinators
            ExecK => {
                if let Exec(top) = mach.pop(&Type::Exec) {
                    let _ = mach.pop(&Type::Exec);
                    mach.push(Exec(top))
                }
            }
            ExecS => {
                if let (Exec(a), Exec(b), Exec(c)) = (
                    mach.pop(&Type::Exec),
                    mach.pop(&Type::Exec),
                    mach.pop(&Type::Exec),
                ) {
                    mach.push(Exec(List(vec![b, c.clone()])));
                    mach.push(Exec(c));
                    mach.push(Exec(a));
                }
            }
            ExecY => {
                if let Exec(a) = mach.pop(&Type::Exec) {
                    if let ExecY = a {
                        // infinite loop detected.
                        log::trace!("Infinite loop detected. Skipping")
                    } else {
                        mach.push(Exec(List(vec![ExecY, a.clone()])));
                        mach.push(Exec(a));
                    }
                }
            }

            CodeQuote => {
                if let Exec(op) = mach.pop(&Type::Exec) {
                    mach.push(Code(op))
                }
            }
            CodeDoAll => {
                while let Some(Code(op)) = mach.pop_opt(&Type::Code) {
                    mach.push(Exec(op))
                }
            }
            ExecIf => {
                // If the top of the Bool stack is `false`, then skip the next
                // instruction in the Exec stack.
                if let Bool(false) = mach.pop(&Type::Bool) {
                    let _ = mach.pop(&Type::Exec);
                }
            }

            // Generic Operations
            Eq(t) => {
                if let (Some(a), Some(b)) = (mach.pop_opt(t), mach.pop_opt(t)) {
                    mach.push(Bool(a == b))
                }
            }
            Rot(t) => {
                if let (Some(a), Some(b), Some(c)) =
                    (mach.pop_opt(t), mach.pop_opt(t), mach.pop_opt(t))
                {
                    mach.push(a);
                    mach.push(c);
                    mach.push(b);
                }
            }
            Swap(t) => {
                if let (Some(a), Some(b)) = (mach.pop_opt(t), mach.pop_opt(t)) {
                    mach.push(a);
                    mach.push(b);
                }
            }
            Drop(t) => {
                let _ = mach.pop_opt(t);
            }
            Dup(t) => {
                if let Some(a) = mach.pop_opt(t) {
                    mach.push(a.clone());
                    mach.push(a);
                }
            }

            // Boolean Operations
            BoolConst(a) => mach.push(Bool(*a)),
            BoolAnd => {
                if let (Bool(a), Bool(b)) = (mach.pop(&Type::Bool), mach.pop(&Type::Bool)) {
                    let res = Bool(a && b);
                    mach.push(res)
                }
            }
            BoolOr => {
                if let (Bool(a), Bool(b)) = (mach.pop(&Type::Bool), mach.pop(&Type::Bool)) {
                    mach.push(Bool(a || b))
                }
            }
            BoolNot => {
                if let Bool(a) = mach.pop(&Type::Bool) {
                    mach.push(Bool(!a))
                }
            }

            // Byte buffer operations
            BufConst(b) => mach.push(Buf(b.clone())),
            BufAt => {
                if let Word(a) = mach.pop(&Type::Word) {
                    mach.push(Buf(Buffer::Pointer(a)))
                }
            }
            BufLen => {
                if let Buf(a) = mach.pop(&Type::Buf) {
                    let len = match a {
                        Buffer::Vector(v) => v.len(),
                        Buffer::Pointer(p) => {
                            let memory = get_static_memory_image();
                            if let Some(buf) = memory.try_dereference(p, None) {
                                buf.len()
                            } else {
                                0
                            }
                        }
                    };
                    mach.push(Word(len as u64))
                }
            }
            BufHead => {
                if let (Buf(buf), Word(n)) = (mach.pop(&Type::Buf), mach.pop(&Type::Word)) {
                    match buf {
                        Buffer::Vector(v) => {
                            if v.len() > 0 {
                                let n = n as usize % v.len();
                                mach.push(Buf(Buffer::Vector(v[0..n].to_vec())))
                            }
                        }
                        Buffer::Pointer(p) => {
                            mach.push(Buf(Buffer::Pointer(p))) // Not much to do here. It's awkward. maybe refactor this later
                        }
                    }
                }
            }
            BufIndex => {
                if let (Buf(buf), Word(i)) = (mach.pop(&Type::Buf), mach.pop(&Type::Word)) {
                    match buf {
                        Buffer::Vector(v) => {
                            if v.len() > 0 {
                                let i = i as usize;
                                let b = v[i % v.len()];
                                mach.push(Word(b as u64));
                            }
                        }
                        Buffer::Pointer(p) => {
                            let memory = get_static_memory_image();
                            if let Some(slice) = memory.try_dereference(p, None) {
                                let i = i as usize;
                                let b = slice[i % slice.len()];
                                mach.push(Word(b as u64));
                            }
                        }
                    }
                }
            }
            BufPack => {
                if let Buf(b) = mach.pop(&Type::Buf) {
                    let memory = loader::get_static_memory_image();
                    match b {
                        Buffer::Pointer(p) => {
                            if let Some(slice) = memory.try_dereference(p, None) {
                                if let Some(n) =
                                    read_integer(slice, memory.endian, memory.word_size)
                                {
                                    mach.push(Word(n));
                                    mach.push(Buf(Buffer::Pointer(p + memory.word_size as u64)))
                                }
                            }
                        }
                        Buffer::Vector(v) => {
                            if let Some(n) = read_integer(&v, memory.endian, memory.word_size) {
                                mach.push(Word(n));
                                if v.len() > memory.word_size {
                                    mach.push(Buf(Buffer::Vector(v[memory.word_size..].to_vec())))
                                }
                            }
                        }
                    }
                }
            }
            BufTail => {
                if let (Buf(b), Word(n)) = (mach.pop(&Type::Buf), mach.pop(&Type::Word)) {
                    match b {
                        Buffer::Vector(v) => {
                            if v.len() > 0 {
                                let n = n as usize % v.len();
                                mach.push(Buf(Buffer::Vector(v[n..].to_vec())))
                            }
                        }
                        Buffer::Pointer(p) => mach.push(Buf(Buffer::Pointer(p + n))),
                    }
                }
            }
            WordDisRegRead => {
                if let Word(w) = mach.pop(&Type::Word) {
                    let memory = loader::get_static_memory_image();
                    if let Some(buf) = memory.try_dereference(w, None) {
                        if let Some(disassembler) = memory.disasm.as_ref() {
                            if let Ok(insts) = disassembler.disas(buf, 0, Some(1)) {
                                for inst in insts.iter() {
                                    if let Ok(details) = disassembler.insn_detail(&inst) {
                                        for reg in details.regs_read() {
                                            let n = reg.0 as u64;
                                            mach.push(Word(n))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            WordDisRegWrite => {
                if let Word(w) = mach.pop(&Type::Word) {
                    let memory = loader::get_static_memory_image();
                    if let Some(buf) = memory.try_dereference(w, None) {
                        if let Some(disassembler) = memory.disasm.as_ref() {
                            if let Ok(insts) = disassembler.disas(buf, 0, Some(1)) {
                                for inst in insts.iter() {
                                    if let Ok(details) = disassembler.insn_detail(&inst) {
                                        for reg in details.regs_write() {
                                            let n = reg.0 as u64;
                                            mach.push(Word(n))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            WordDisInstId => {
                if let Word(w) = mach.pop(&Type::Word) {
                    let memory = loader::get_static_memory_image();
                    if let Some(buf) = memory.try_dereference(w, None) {
                        if let Some(disassembler) = memory.disasm.as_ref() {
                            if let Ok(insts) = disassembler.disas(buf, 0, Some(1)) {
                                for inst in insts.iter() {
                                    mach.push(Word(inst.id().0 as u64))
                                }
                            }
                        }
                    }
                }
            }
            BufDisRegRead => {
                if let Buf(buf) = mach.pop(&Type::Buf) {
                    let memory = loader::get_static_memory_image();
                    if let Some(disassembler) = memory.disasm.as_ref() {
                        if let Ok(insts) = match buf {
                            Buffer::Vector(v) => disassembler.disas(&v, 0, Some(1)),
                            Buffer::Pointer(p) => {
                                if let Some(slice) = memory.try_dereference(p, None) {
                                    disassembler.disas(slice, 0, Some(1))
                                } else {
                                    disassembler.disas(&vec![], 0, Some(1))
                                }
                            }
                        } {
                            for inst in insts.iter() {
                                if let Ok(details) = disassembler.insn_detail(&inst) {
                                    for reg in details.regs_read() {
                                        let n = reg.0 as u64;
                                        mach.push(Word(n))
                                    }
                                }
                            }
                        }
                    }
                }
            }
            BufDisRegWrite => {
                if let Buf(buf) = mach.pop(&Type::Buf) {
                    let memory = loader::get_static_memory_image();
                    if let Some(disassembler) = memory.disasm.as_ref() {
                        if let Ok(insts) = match buf {
                            Buffer::Vector(v) => disassembler.disas(&v, 0, Some(1)),
                            Buffer::Pointer(p) => {
                                if let Some(slice) = memory.try_dereference(p, None) {
                                    disassembler.disas(slice, 0, Some(1))
                                } else {
                                    disassembler.disas(&vec![], 0, Some(1))
                                }
                            }
                        } {
                            for inst in insts.iter() {
                                if let Ok(details) = disassembler.insn_detail(&inst) {
                                    for reg in details.regs_write() {
                                        let n = reg.0 as u64;
                                        mach.push(Word(n))
                                    }
                                }
                            }
                        }
                    }
                }
            }
            BufDisInstId => {
                if let Buf(buf) = mach.pop(&Type::Buf) {
                    let memory = loader::get_static_memory_image();
                    if let Some(disassembler) = memory.disasm.as_ref() {
                        match buf {
                            Buffer::Pointer(p) => {
                                if let Some(slice) = memory.try_dereference(p, None) {
                                    if let Ok(insts) = disassembler.disas(slice, 0, Some(1)) {
                                        for inst in insts.iter() {
                                            mach.push(Word(inst.id().0 as u64))
                                        }
                                    }
                                }
                            }
                            Buffer::Vector(v) => {
                                if let Ok(insts) = disassembler.disas(&v, 0, Some(1)) {
                                    for inst in insts.iter() {
                                        mach.push(Word(inst.id().0 as u64))
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Word (integer/address) Operations
            WordConst(w) => mach.push(Word(*w)),
            WordShl => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    let b = b as u32 % 64;
                    let (res, _) = a.overflowing_shl(b);
                    mach.push(Word(res))
                }
            }
            WordShr => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    let b = b as u32 % 64;
                    let (res, _) = a.overflowing_shr(b);
                    mach.push(Word(res))
                }
            }
            WordLess => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    mach.push(Bool(a < b))
                }
            }
            WordEqual => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    mach.push(Bool(a == b))
                }
            }
            WordAdd => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    mach.push(Word(a.wrapping_add(b)))
                }
            }
            WordSub => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    mach.push(Word(a.wrapping_sub(b)))
                }
            }
            WordAnd => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    mach.push(Word(a & b))
                }
            }
            WordOr => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    mach.push(Word(a | b))
                }
            }
            WordXor => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    mach.push(Word(a ^ b))
                }
            }
            WordMul => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    mach.push(Word(a.wrapping_mul(b)))
                }
            }
            WordDiv => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    if b != 0 {
                        mach.push(Word(a / b))
                    }
                }
            }
            WordMod => {
                if let (Word(a), Word(b)) = (mach.pop(&Type::Word), mach.pop(&Type::Word)) {
                    if b != 0 {
                        mach.push(Word(a % b))
                    }
                }
            }
            WordIncr => {
                if let Word(a) = mach.pop(&Type::Word) {
                    mach.push(Word(a.wrapping_add(1)))
                }
            }
            WordDecr => {
                if let Word(a) = mach.pop(&Type::Word) {
                    mach.push(Word(a.wrapping_sub(1)))
                }
            }
            WordOnes => {
                if let Word(a) = mach.pop(&Type::Word) {
                    mach.push(Word(a.count_ones() as u64))
                }
            }
            WordDeref => {
                if let Word(a) = mach.pop(&Type::Word) {
                    let memory = get_static_memory_image();
                    if let Some(x) = memory.try_dereference(a, None) {
                        if let Some(w) = read_integer(x, memory.endian, memory.word_size) {
                            mach.push(Word(w))
                        }
                    }
                }
            }
            WordSearch => {
                if let Word(a) = mach.pop(&Type::Word) {
                    let memory = get_static_memory_image();
                    let mut bytes = vec![0; memory.word_size];
                    write_integer(memory.endian, memory.word_size, a, &mut bytes);
                    if let Some(addr) = memory.seek_all_segs(&bytes, None) {
                        mach.push(Word(addr))
                    }
                }
            }
            WordReadable => {
                if let Word(a) = mach.pop(&Type::Word) {
                    let memory = get_static_memory_image();
                    let perm = memory.perm_of_addr(a).unwrap_or(Perms::NONE);
                    mach.push(Bool(perm.intersects(Perms::READ)))
                }
            }
            WordWriteable => {
                if let Word(a) = mach.pop(&Type::Word) {
                    let memory = get_static_memory_image();
                    let perm = memory.perm_of_addr(a).unwrap_or(Perms::NONE);
                    mach.push(Bool(perm.intersects(Perms::WRITE)))
                }
            }
            WordExecutable => {
                if let Word(a) = mach.pop(&Type::Word) {
                    let memory = get_static_memory_image();
                    let perm = memory.perm_of_addr(a).unwrap_or(Perms::NONE);
                    mach.push(Bool(perm.intersects(Perms::EXEC)))
                }
            }
            FloatLog => {
                if let Float(a) = mach.pop(&Type::Float) {
                    mach.push(Float(f64::from_bits(a).ln().to_bits()))
                }
            }
            FloatCos => {
                if let Float(a) = mach.pop(&Type::Float) {
                    mach.push(Float(f64::from_bits(a).cos().to_bits()))
                }
            }
            FloatSin => {
                if let Float(a) = mach.pop(&Type::Float) {
                    mach.push(Float(f64::from_bits(a).sin().to_bits()))
                }
            }
            FloatTanh => {
                if let Float(a) = mach.pop(&Type::Float) {
                    mach.push(Float(f64::from_bits(a).tanh().to_bits()))
                }
            }
            FloatTan => {
                if let Float(a) = mach.pop(&Type::Float) {
                    mach.push(Float(f64::from_bits(a).tan().to_bits()))
                }
            }
            FloatConst(n) => mach.push(Float(*n)),
            FloatLess => {
                if let (Float(a), Float(b)) = (mach.pop(&Type::Float), mach.pop(&Type::Float)) {
                    let a = f64::from_bits(a);
                    let b = f64::from_bits(b);
                    mach.push(Bool(a < b))
                }
            }

            FloatToWord => {
                if let Float(a) = mach.pop(&Type::Float) {
                    mach.push(Word(f64::from_bits(a) as u64))
                }
            }

            WordToFloat => {
                if let Word(a) = mach.pop(&Type::Word) {
                    mach.push(Float(a))
                }
            }
            WordToGadget => {
                if let Word(a) = mach.pop(&Type::Word) {
                    mach.push(Gadget(a))
                }
            }
            GadgetToWord => {
                if let Gadget(a) = mach.pop(&Type::Gadget) {
                    mach.push(Word(a))
                }
            }

            // Perform a very much approximate evaluation of the expression, by unwrapping its
            // arguments, and roughly translating the operation
            ExprEval => {
                // this one is a bit complex.
                if let Expression(expr) = mach.pop(&Type::Expression) {
                    use il::Expression as E;
                    if let E::Constant(c) = expr {
                        if let Some(n) = c.value_u64() {
                            mach.push(Word(n))
                        }
                    } else {
                        let (op, a, b) = match expr {
                            E::Add(a, b) => (WordAdd, Some(a), Some(b)),
                            E::Sub(a, b) => (WordSub, Some(a), Some(b)),
                            E::Mul(a, b) => (WordMul, Some(a), Some(b)),
                            E::Divu(a, b) => (WordDiv, Some(a), Some(b)),
                            E::Modu(a, b) => (WordMod, Some(a), Some(b)),
                            E::Divs(a, b) => (WordDiv, Some(a), Some(b)),
                            E::Mods(a, b) => (WordMod, Some(a), Some(b)),
                            E::And(a, b) => (WordAnd, Some(a), Some(b)),
                            E::Or(a, b) => (WordOr, Some(a), Some(b)),
                            E::Xor(a, b) => (WordXor, Some(a), Some(b)),
                            E::Cmpeq(a, b) => (Eq(Type::Word), Some(a), Some(b)),
                            E::Cmpneq(a, b) => (Eq(Type::Word), Some(a), Some(b)),
                            E::Cmplts(a, b) => (WordLess, Some(a), Some(b)),
                            E::Cmpltu(a, b) => (WordLess, Some(a), Some(b)),
                            E::Ite(a, b, _) => (ExecIf, Some(a), Some(b)),
                            // scalars and constants
                            _ => (Nop, None, None),
                        };
                        mach.push(Code(op));
                        if let Some(a) = a {
                            mach.push(Expression(*a))
                        }
                        if let Some(b) = b {
                            mach.push(Expression(*b))
                        }
                    }
                }
            }
            // // Given a Function and a CFG index, get the block at that index
            FuncBlocks => {
                if let Function(f) = mach.pop(&Type::Function) {
                    let cfg = f.control_flow_graph();
                    for block in cfg.blocks() {
                        mach.push(Block(block));
                    }
                }
            }
            // FuncEntry => {
            //     if let Function(f) = mach.pop(&Type::Function) {
            //         if let Some(i) = f.control_flow_graph().entry() {
            //             if let Ok(block) = f.block(i) {
            //                 mach.push(Block(block));
            //             }
            //         }
            //     }
            // }
            // FuncExit => {
            //     if let Function(f) = mach.pop(&Type::Function) {
            //         let cfg = f.control_flow_graph();
            //         if let Some(i) = cfg.exit() {
            //             if let Ok(block) = f.block(i) {
            //                 let tail_index = block.index();
            //                 if let Ok(indices) = cfg.predecessor_indices(tail_index) {
            //                     for head_index in indices {
            //                         if let Ok(block) = cfg.block(head_index) {
            //                             mach.push(Block(block));
            //                         }
            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }
            FuncNamed(name) => {
                let memory = get_static_memory_image();
                let program = memory
                    .il_program
                    .as_ref()
                    .expect("No IR program structure!");
                let func = program.function_by_name(name).expect("Bad function name");
                mach.push(Function(func));
            }

            FuncAddr => {
                if let Function(func) = mach.pop(&Type::Function) {
                    let addr = func.address();
                    mach.push(Word(addr));
                }
            }
            //
            // Not very useful, and ill-conceived. each constant analyser corresponds to a sinle
            // proram location. fuck i miss my  key. my jee key. arhhh.

            // FuncConstants => {
            //     if let Function(func) = mach.pop(&Type::Function) {
            //         if let Ok(constant_analyser) = falcon::analysis::constants::constants(func) {
            //             for c in constant_analyser.values() {
            //                 println!("Got constant analyser for {:?}", Function(func));
            //                 while let Some(Scalar(s)) = mach.pop_opt(&Type::Scalar) {
            //                     println!("Tryin to analyse scalar {:?}", s);
            //                     if let Some(v) = c.scalar(&s) {
            //                         if let Some(w) = v.value_u64() {
            //                             println!("Resolved constant! {:?} -> {}", v, w);
            //                             mach.push(Word(w));
            //                         }
            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }
            BlockAddr => {
                if let Block(a) = mach.pop(&Type::Block) {
                    if let Some(addr) = a.address() {
                        mach.push(Gadget(addr));
                    }
                }
            }
            BlockInsts => {
                if let Block(a) = mach.pop(&Type::Block) {
                    for inst in a.instructions() {
                        mach.push(Instruction(inst))
                    }
                }
            }
            // If the block on the top of the block stack has any successors, retrieve them
            // and push them onto the block stack. If any of the edges to the successors are
            // conditional, push the conditional expression to the expression stack.
            BlockSuccs => {
                if let (Function(f), Block(b)) = (mach.pop(&Type::Function), mach.pop(&Type::Block))
                {
                    let head_index = b.index();
                    let cfg = f.control_flow_graph();
                    if let Ok(indices) = cfg.successor_indices(head_index) {
                        // println!("{:?} has {} successors", Block(b), indices.len());
                        for tail_index in indices {
                            if let Ok(block) = cfg.block(tail_index) {
                                if let Ok(edge) = cfg.edge(head_index, tail_index) {
                                    if let Some(condition) = edge.condition() {
                                        mach.push(Expression(condition.clone()))
                                    }
                                }
                                // println!("Pushing successor {:?}", Block(block));
                                mach.push(Block(block));
                            }
                        }
                    }
                }
            }
            BlockPreds => {
                if let (Function(f), Block(b)) = (mach.pop(&Type::Function), mach.pop(&Type::Block))
                {
                    let tail_index = b.index();
                    let cfg = f.control_flow_graph();
                    if let Ok(indices) = cfg.predecessor_indices(tail_index) {
                        for head_index in indices {
                            if let Ok(block) = cfg.block(head_index) {
                                if let Ok(edge) = cfg.edge(head_index, tail_index) {
                                    if let Some(condition) = edge.condition() {
                                        mach.push(Expression(condition.clone()))
                                    }
                                }
                                mach.push(Block(block));
                            }
                        }
                    }
                }
            }

            InstIsLoad => {
                if let Instruction(a) = mach.pop(&Type::Instruction) {
                    if let il::Operation::Load { dst, index } = a.operation() {
                        mach.push(Expression(index.clone()));
                        mach.push(Scalar(dst.clone()));
                    }
                    mach.push(Bool(a.is_load()))
                }
            }
            InstIsStore => {
                if let Instruction(a) = mach.pop(&Type::Instruction) {
                    if let il::Operation::Store { index, src } = a.operation() {
                        mach.push(Expression(index.clone()));
                        mach.push(Expression(src.clone()));
                    }
                    mach.push(Bool(a.is_store()));
                }
            }
            InstIsBranch => {
                if let Instruction(a) = mach.pop(&Type::Instruction) {
                    if let il::Operation::Branch { target } = a.operation() {
                        mach.push(Expression(target.clone()));
                    }
                    mach.push(Bool(a.is_branch()));
                }
            }
            InstAddr => {
                if let Instruction(a) = mach.pop(&Type::Instruction) {
                    if let Some(addr) = a.address() {
                        mach.push(Gadget(addr))
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    Word,
    Buf,
    // Not using byte slices yet. but we could try later.
    Bool,
    Exec,
    Code,
    Perm,
    None,
    Float,
    Gadget,
    // Falcon-specific
    Scalar,
    Expression,
    //Operation,
    Instruction,
    Block,
    Function,
}

impl From<&Val> for Type {
    fn from(v: &Val) -> Self {
        match v {
            Val::Word(_) => Type::Word,
            Val::Gadget(_) => Type::Gadget,
            Val::Buf(_) => Type::Buf,
            Val::Bool(_) => Type::Bool,
            Val::Exec(_) => Type::Exec,
            Val::Float(_) => Type::Float,
            Val::Code(_) => Type::Code,
            Val::Scalar(_) => Type::Scalar,
            Val::Expression(_) => Type::Expression,
            //Val::Operation(_) => Type::Operation,
            Val::Instruction(_) => Type::Instruction,
            Val::Block(_) => Type::Block,
            Val::Function(_) => Type::Function,
            Val::Null => Type::None,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum Val {
    Null,
    Word(u64),
    Buf(Buffer),
    Bool(bool),
    Exec(Op),
    Code(Op),
    Float(u64),
    Gadget(u64),
    // Falcon
    Scalar(il::Scalar),
    Expression(il::Expression),
    //Operation(&'static il::Operation),
    Instruction(&'static il::Instruction),
    Block(&'static il::Block),
    Function(&'static il::Function),
}

impl fmt::Debug for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Val::*;
        match self {
            Null => write!(f, "Null"),
            Word(w) => write!(f, "Word(0x{:x})", w),
            Buf(b) => {
                write!(f, "Buf({:x?})", b)
                // let i = 16.min(b.len());
                // let ellipsis = if i < b.len() { "..." } else { "" };
                // write!(f, "Buf({:x?}{})", &b[..i], ellipsis)
            }
            Bool(b) => write!(f, "Bool({:?})", b),
            Exec(op) => write!(f, "Exec({:?})", op),
            Code(op) => write!(f, "Code({:?})", op),
            Float(bits) => write!(f, "Float({})", f64::from_bits(*bits)),
            Gadget(w) => write!(f, "Gadget(0x{:x})", w),
            Scalar(s) => write!(f, "Scalar({:?})", s),
            Expression(e) => write!(f, "Expression({})", e),
            Instruction(inst) => write!(f, "Instruction({})", inst),
            Block(b) => write!(f, "Block(at 0x{:x})", b.address().unwrap_or(0)),
            Function(func) => write!(f, "Function(named {})", func.name()),
        }
    }
}

// TODO: disassembly aware instructions.
//

impl Val {
    pub fn unwrap_word(self) -> Option<u64> {
        match self {
            Self::Word(w) | Self::Gadget(w) => Some(w),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MachineState {
    stacks: HashMap<Type, Stack<Val>>,
    counter: usize,
}

// TODO try optimizing by getting rid of the hashmap in favour of just
// using struct fields
impl MachineState {
    pub fn load_args(&mut self, args: &[Val]) {
        for arg in args {
            self.push(arg.clone())
        }
    }

    pub fn flush(&mut self) {
        self.stacks = HashMap::new();
        self.stacks.insert(Type::Block, vec![]);
        self.stacks.insert(Type::Bool, vec![]);
        self.stacks.insert(Type::Buf, vec![]);
        self.stacks.insert(Type::Code, vec![]);
        self.stacks.insert(Type::Exec, vec![]);
        self.stacks.insert(Type::Expression, vec![]);
        self.stacks.insert(Type::Float, vec![]);
        self.stacks.insert(Type::Function, vec![]);
        self.stacks.insert(Type::Gadget, vec![]);
        self.stacks.insert(Type::Instruction, vec![]);
        self.stacks.insert(Type::Scalar, vec![]);
        self.stacks.insert(Type::Word, vec![]);
        self.counter = 0;
    }

    // the only reason for using Val::Null is to make the code
    // a bit more ergonomic
    pub fn pop_opt(&mut self, t: &Type) -> Option<Val> {
        self.stacks.get_mut(t).expect("missing stack").pop()
    }

    pub fn pop(&mut self, t: &Type) -> Val {
        self.pop_opt(t).unwrap_or(Val::Null)
    }

    pub fn push(&mut self, val: Val) {
        let s = Type::from(&val);
        if let Type::None = s {
            return;
        }
        self.stacks
            .get_mut(&s)
            .expect("missing output stack")
            .push(val)
    }

    pub fn exec(&mut self, code: &[Op], args: &[Val], max_steps: usize) -> Vec<u64> {
        self.flush();
        self.load_args(args);
        // Load the exec stack
        for op in code {
            self.push(Val::Exec(op.clone()))
        }

        while let Some(Val::Exec(op)) = self.pop_opt(&Type::Exec) {
            log::trace!("[{}] {:x?}", self.counter, op);
            self.counter += 1;
            if self.counter >= max_steps {
                break;
            }

            op.eval(self)
        }

        log::trace!("Completed execution. Machine state: {:#?}", self);
        // first, take the explicitly-marked gadgets.
        // ensure that this list begins with an executable.

        let mut payload = self
            .stacks
            .get_mut(&Type::Word)
            .unwrap()
            .drain(..)
            .filter_map(Val::unwrap_word)
            .collect::<Vec<u64>>();

        let gadgets = self
            .stacks
            .get_mut(&Type::Gadget)
            .unwrap()
            .drain(..)
            .filter_map(Val::unwrap_word);
        payload.extend(gadgets);

        let memory = get_static_memory_image();
        while !payload
            .last()
            .and_then(|a| memory.perm_of_addr(*a))
            .map(|p| p.intersects(Perms::EXEC))
            .unwrap_or(false)
        {
            if payload.pop().is_none() {
                break;
            };
        }
        payload.reverse();

        log::trace!("Payload: {:#x?}", payload);
        payload
    }
}

// We want a way to prime the machine with the specification of a problem.
// For example, we should be able to take a register pattern, and transform it
// into a set of arguments to be fed into the machine.
fn register_pattern_to_push_args(rp: &RegisterPattern) -> Vec<Val> {
    // it doesn't exactly matter which register is which, I think, so long
    // as they're handled in a predictable order. One order is as good as
    // another, from this perspective, so we'll use the simplest: alphabetical,
    // by register name.
    let mut args = Vec::new();
    for (_reg, rval) in rp.0.iter().sorted_by_key(|p| p.0) {
        for w in rval.vals.iter() {
            args.push(Val::Word(*w));
        }
        // alternately, to do this in an easier way, if deref > 0, then
        // first search memory for an occurrence of val, and then return the
        // address.
        args.push(Val::Word(rval.deref as u64));
    }
    args
}

pub mod creature {
    use std::fmt;
    use std::hash::{Hash, Hasher};

    use rand::thread_rng;

    use crate::emulator::profiler::{HasProfile, Profile};
    use crate::evolution::{Genome, LinearChromosome, Mutation, Phenome};
    use crate::roper::Fitness;
    use crate::util;
    use crate::util::random::hash_seed_rng;

    use super::*;

    #[derive(Clone, Debug, Copy, Hash, Serialize, Deserialize)]
    pub enum OpMutation {
        RandomOp,
    }

    impl Mutation for OpMutation {
        type Allele = Op;

        // TODO: should we have a pointer to the config here?
        // we might want to take certain parameters into consideration, like the literal rate
        fn mutate_point(allele: &mut Self::Allele, config: &Config) -> Self {
            let mut rng = thread_rng();
            let new_allele = random_ops(&mut rng, config).pop().unwrap();
            *allele = new_allele;
            OpMutation::RandomOp
        }
    }

    #[derive(Clone, Serialize)]
    pub struct Creature {
        pub chromosome: LinearChromosome<Op, OpMutation>,
        pub tag: u64,
        // TODO: this should become a hashmap associating problems with payloads
        pub payloads: Vec<Vec<u64>>,
        // But then we need some way to map the problems to the profiles. not just
        // flat vecs. I think we may need to refactor the profile struct, which could
        // get a bit messy. For the best, though.
        // We don't need to hold the problems themselves, here. An index into a problem
        // table held in Config would be just fine. We can always get a pointer to Config
        // in scope.
        pub profile: Option<Profile>,
        #[serde(borrow)]
        pub fitness: Option<Fitness<'static>>,
        pub front: Option<usize>,
        pub num_offspring: usize,
        pub native_island: usize,
        pub description: Option<String>,
    }

    impl HasProfile for Creature {
        fn profile(&self) -> Option<&Profile> {
            self.profile.as_ref()
        }

        fn add_profile(&mut self, profile: Profile) {
            if let Some(ref mut p) = self.profile {
                p.absorb(profile)
            } else {
                self.profile = Some(profile)
            }
        }

        fn set_profile(&mut self, profile: Profile) {
            self.profile = Some(profile)
        }
    }

    impl Hash for Creature {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.tag.hash(state)
        }
    }

    impl Creature {}

    impl Genome for Creature {
        type Allele = Op;

        fn generation(&self) -> usize {
            self.chromosome.generation
        }

        fn num_offspring(&self) -> usize {
            self.num_offspring
        }

        fn chromosome(&self) -> &[Self::Allele] {
            &self.chromosome.chromosome
        }

        fn chromosome_mut(&mut self) -> &mut [Self::Allele] {
            &mut self.chromosome.chromosome
        }

        fn random<H: Hash>(config: &Config, salt: H) -> Self
        where
            Self: Sized,
        {
            // First, let's get some information from the lifted program
            let mut rng = hash_seed_rng(&salt);
            let length = rng.gen_range(config.push_vm.min_len, config.push_vm.max_len);
            let ops = random_ops(&mut rng, config);

            Self {
                chromosome: LinearChromosome {
                    chromosome: ops,
                    mutations: vec![None; length],
                    parentage: vec![],
                    parent_names: vec![],
                    name: util::name::random(4, rng.gen::<u64>()),
                    generation: 0,
                },
                tag: rng.gen::<u64>(),
                payloads: vec![],
                profile: None,
                fitness: None,
                front: None,
                num_offspring: 0,
                native_island: config.island_id,
                description: None,
            }
        }

        fn crossover(mates: &[&Self], config: &Config) -> Self
        where
            Self: Sized,
        {
            let parents = mates
                .iter()
                .map(|x| &x.chromosome)
                .collect::<Vec<&LinearChromosome<_, _>>>();
            let chromosome = LinearChromosome::crossover(&parents, config);
            Self {
                chromosome,
                tag: thread_rng().gen::<u64>(),
                payloads: vec![],
                profile: None,
                fitness: None,
                front: None,
                num_offspring: 0,
                native_island: 0,
                description: None,
            }
        }

        fn mutate(&mut self, config: &Config) {
            self.chromosome.mutate(config)
        }

        fn incr_num_offspring(&mut self, _n: usize) {
            self.num_offspring += 1
        }
    }

    impl Phenome for Creature {
        type Fitness = Fitness<'static>;
        type Problem = ();

        fn generate_description(&mut self) {
            self.description = Some(format!("{:#?}", self))
        }

        fn fitness(&self) -> Option<&Self::Fitness> {
            self.fitness.as_ref()
        }

        fn scalar_fitness(&self, weighting: &str) -> Option<f64> {
            self.fitness
                .as_ref()
                .map(|f| f.scalar_with_expression(weighting))
        }

        fn set_fitness(&mut self, f: Self::Fitness) {
            self.fitness = Some(f)
        }

        fn tag(&self) -> u64 {
            self.tag
        }

        fn set_tag(&mut self, tag: u64) {
            self.tag = tag
        }

        fn answers(&self) -> Option<&Vec<Self::Problem>> {
            unimplemented!()
        }

        fn store_answers(&mut self, _results: Vec<Self::Problem>) {
            unimplemented!()
        }

        fn is_goal_reached(&self, config: &Config) -> bool {
            self.scalar_fitness(&config.fitness.priority())
                .map(|p| p - config.fitness.target <= std::f64::EPSILON)
                .unwrap_or(false)
        }
    }

    impl fmt::Debug for Creature {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(f, "Native island {}", self.native_island)?;
            writeln!(f, "{:?}", self.chromosome)?;
            writeln!(f, "{} payloads", self.payloads.len())?;

            for (p_num, payload) in self.payloads.iter().enumerate() {
                writeln!(f, "payload {}, of length {}", p_num, payload.len())?;
                let memory = get_static_memory_image();
                for (i, w) in payload.iter().enumerate() {
                    let perms = memory
                        .perm_of_addr(*w)
                        .map(|p| format!(" ({:?})", p))
                        .unwrap_or_else(String::new);
                    let executed = self
                        .profile
                        .as_ref()
                        .map(|p| p.times_executed(*w) > 0)
                        .unwrap_or(false);
                    writeln!(
                        f,
                        "[{p_num}:{i}] 0x{word:010x}{perms}{executed}",
                        p_num = p_num,
                        i = i,
                        word = w,
                        perms = perms,
                        executed = if executed { " *" } else { "" },
                    )?;
                }
            }
            if let Some(ref profile) = self.profile {
                for (i, path) in profile.disas_paths().enumerate() {
                    writeln!(f, "Trace for payload {}:", i)?;
                    writeln!(f, "{}", path)?;
                }
                for (i, state) in profile.registers.iter().enumerate() {
                    writeln!(
                        f,
                        "\nSpidered register state for payload {}:\n{:?}",
                        i, state
                    )?;
                }
                writeln!(f, "CPU Error code(s): {:?}", profile.cpu_errors)?;
            }
            writeln!(f, "Fitness: {:#?}", self.fitness())?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod test {
    use unicorn::{Arch, Mode};

    use crate::configure::RoperConfig;

    use super::*;

    #[test]
    fn test_random_ops() {
        // first initialize things
        let roper_config = RoperConfig {
            gadget_file: None,
            output_registers: vec![],
            randomize_registers: false,
            soup: None,
            soup_size: Some(1024),
            arch: Arch::X86,
            mode: Mode::MODE_64,
            num_workers: 0,
            num_emulators: 0,
            wait_limit: 0,
            max_emu_steps: None,
            millisecond_timeout: None,
            record_basic_blocks: false,
            record_memory_writes: false,
            emulator_stack_size: 0,
            binary_path: "./binaries/X86/MODE_64/nano".to_string(),
            ld_paths: None,
            bad_bytes: None,
            ..Default::default()
        };
        let mut config = Config::default();
        config.roper = roper_config;
        config.push_vm = crate::configure::PushVm {
            max_steps: 100,
            min_len: 10,
            max_len: 100,
            literal_rate: 0.3,
        };
        loader::falcon_loader::load_from_path(&mut config, true).expect("failed to load");
        crate::roper::init_soup(&mut config).expect("Failed to init soup");
        crate::logger::init("test");
        println!("Loading, linking, and lifting...");
        let ops = random_ops(&mut rand::thread_rng(), &config);
        println!("{:#?}", ops);
        let mut machine = MachineState::default();
        let args = vec![];
        let res = machine.exec(&ops, &args, 1000);
        println!("Machine state: {:#?}", machine);
        println!("Result: {:#x?}", res);
    }
}
