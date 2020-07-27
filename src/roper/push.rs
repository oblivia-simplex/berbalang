use std::fmt;

use bitflags::_core::fmt::Formatter;
use capstone::RegId;
use falcon::il;
use hashbrown::HashMap;
use rand::prelude::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::configure::Config;
use crate::emulator::loader;
use crate::emulator::loader::get_static_memory_image;
use crate::util::architecture::{read_integer, write_integer, Perms};

pub type Stack<T> = Vec<T>;

pub type Input = Type;
pub type Output = Type;

// TODO: define a distribution over these ops. WordConst should comprise about half
// of any genome, I think, since without that, there's no ROP chain.

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
    WordOnes,
    WordShl,
    WordShr,

    // Treating the Word as an address
    WordDeref,
    WordSearch,
    //WordReadRegs,
    //WordWriteRegs,
    // TODO: there should be a way to map addresses to IL CFG Nodes
    WordToFloat,

    WordReadable,
    WordWriteable,
    WordExecutable,

    WordToGadget,
    GadgetToWord,

    //BufConst(&'static [u8]),
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
    FuncBlock,
    FuncEntry,
    FuncExit,
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

static NON_CONSTANT_OPS: [Op; 60] = [
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
    Op::WordShl,
    Op::WordShr,
    Op::WordDeref,
    Op::WordSearch,
    // Op::WordReadRegs,
    // Op::WordWriteRegs,
    Op::WordToFloat,
    Op::WordReadable,
    Op::WordWriteable,
    Op::WordExecutable,
    Op::WordToGadget,
    Op::GadgetToWord,
    //Op::BufConst(&'static [u8]),
    Op::BufAt,
    Op::BufLen,
    Op::BufIndex,
    Op::BufPack,
    Op::BufHead,
    Op::BufTail,
    // Op::BufSearch,
    Op::BufDisRegRead,
    Op::BufDisRegWrite,
    Op::BufDisInstId,
    // floats need to be encoded as u64 to preserve Op::Hash
    Op::FloatLog,
    Op::FloatSin,
    Op::FloatCos,
    Op::FloatTan,
    Op::FloatTanh,
    Op::FloatToWord,
    Op::FloatLess,
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
    Op::FuncBlock,
    Op::FuncEntry,
    Op::FuncExit,
    Op::FuncAddr,
    Op::ExprEval,
    Op::ExecIf,
    Op::ExecK,
    Op::ExecS,
    Op::ExecY,
    Op::Nop,
];

fn random_ops<R: Rng>(rng: &mut R, literal_rate: f64, count: usize) -> Vec<Op> {
    let mut ops = Vec::new();

    let memory = get_static_memory_image();
    let program = memory.il_program.as_ref().unwrap();
    let function_names: Vec<String> = program
        .functions()
        .into_iter()
        .map(il::Function::name)
        .collect();

    for _ in 0..count {
        if rng.gen_range(0.0, 1.0) < literal_rate {
            match rng.gen_range(0, 4) {
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
                    let addr = memory.random_address(None, rng.gen::<u64>());
                    ops.push(Op::WordConst(addr))
                }
                2 => {
                    // boolean
                    ops.push(Op::BoolConst(rng.gen::<bool>()))
                }
                3 => {
                    // float
                    ops.push(Op::FloatConst(rng.gen::<f64>().to_bits()))
                }
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
                    mach.push(Exec(List(vec![ExecY, a.clone()])));
                    mach.push(Exec(a));
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
            BufAt => {
                if let Word(a) = mach.pop(&Type::Word) {
                    let memory = loader::get_static_memory_image();
                    if let Some(buf) = memory.try_dereference(a, None) {
                        mach.push(Buf(buf))
                    }
                }
            }
            BufLen => {
                if let Buf(a) = mach.pop(&Type::Buf) {
                    mach.push(Word(a.len() as u64))
                }
            }
            BufHead => {
                if let (Buf(buf), Word(n)) = (mach.pop(&Type::Buf), mach.pop(&Type::Word)) {
                    let n = n as usize % buf.len();
                    mach.push(Buf(&buf[0..n]))
                }
            }
            BufIndex => {
                if let (Buf(buf), Word(i)) = (mach.pop(&Type::Buf), mach.pop(&Type::Word)) {
                    let i = i as usize;
                    let b = buf[i % buf.len()];
                    mach.push(Word(b as u64));
                }
            }
            BufPack => {
                if let Buf(b) = mach.pop(&Type::Buf) {
                    let memory = loader::get_static_memory_image();
                    if let Some(n) = read_integer(b, memory.endian, memory.word_size) {
                        mach.push(Word(n))
                    }
                }
            }
            BufTail => {
                if let (Buf(b), Word(n)) = (mach.pop(&Type::Buf), mach.pop(&Type::Word)) {
                    let n = n as usize % b.len();
                    mach.push(Buf(b[..n].as_ref()))
                }
            }
            BufDisRegRead => {
                if let Buf(buf) = mach.pop(&Type::Buf) {
                    let memory = loader::get_static_memory_image();
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
            BufDisRegWrite => {
                if let Buf(buf) = mach.pop(&Type::Buf) {
                    let memory = loader::get_static_memory_image();
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
            BufDisInstId => {
                if let Buf(buf) = mach.pop(&Type::Buf) {
                    let memory = loader::get_static_memory_image();
                    if let Some(disassembler) = memory.disasm.as_ref() {
                        if let Ok(insts) = disassembler.disas(buf, 0, Some(1)) {
                            for inst in insts.iter() {
                                mach.push(Word(inst.id().0 as u64))
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
            // Given a Function and a CFG index, get the block at that index
            FuncBlock => {
                if let (Function(f), Word(i)) = (mach.pop(&Type::Function), mach.pop(&Type::Word)) {
                    let index = i as usize;
                    if let Ok(block) = f.block(index) {
                        mach.push(Block(block));
                    }
                }
            }
            FuncEntry => {
                if let Function(f) = mach.pop(&Type::Function) {
                    if let Some(i) = f.control_flow_graph().entry() {
                        if let Ok(block) = f.block(i) {
                            mach.push(Block(block));
                        }
                    }
                }
            }
            FuncExit => {
                if let Function(f) = mach.pop(&Type::Function) {
                    if let Some(i) = f.control_flow_graph().exit() {
                        if let Ok(block) = f.block(i) {
                            mach.push(Block(block));
                        }
                    }
                }
            }
            FuncAddr => {
                if let Function(f) = mach.pop(&Type::Function) {
                    let addr = f.address();
                    mach.push(Word(addr));
                }
            }
            FuncNamed(name) => {
                let memory = get_static_memory_image();
                if let Some(ref program) = memory.il_program {
                    if let Some(func) = program.function_by_name(name) {
                        mach.push(Function(func))
                    }
                }
            }

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
                        for tail_index in indices {
                            if let Ok(block) = cfg.block(tail_index) {
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
                    mach.push(Bool(a.is_load()))
                }
            }
            InstIsStore => {
                if let Instruction(a) = mach.pop(&Type::Instruction) {
                    mach.push(Bool(a.is_store()))
                }
            }
            InstIsBranch => {
                if let Instruction(a) = mach.pop(&Type::Instruction) {
                    mach.push(Bool(a.is_branch()))
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Val {
    Null,
    Word(u64),
    Buf(&'static [u8]),
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

        log::trace!("Completed execution");
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

        payload
    }
}

pub mod creature {
    use std::fmt;
    use std::hash::{Hash, Hasher};

    use crate::emulator::loader;
    use crate::emulator::profiler::Profile;
    use crate::evolution::{Genome, Phenome};
    use crate::fitness::{HasScalar, MapFit, Weighted};
    use crate::roper::Fitness;
    use crate::util;
    use crate::util::random::hash_seed_rng;

    use super::*;

    #[derive(Clone, Debug, Copy, Hash, Serialize, Deserialize)]
    pub enum Mutation {}

    #[derive(Clone, Serialize, Deserialize, Debug, Default)]
    pub struct Creature {
        pub chromosome: Vec<Op>,
        pub chromosome_parentage: Vec<usize>,
        pub chromosome_mutation: Vec<Option<Mutation>>,
        pub tag: u64,
        pub name: String,
        pub parents: Vec<String>,
        pub generation: usize,
        pub payload: Option<Vec<u64>>,
        pub profile: Option<Profile>,
        #[serde(borrow)]
        pub fitness: Option<Fitness<'static>>,
        pub front: Option<usize>,
        pub num_offspring: usize,
        pub native_island: usize,
        pub description: Option<String>,
    }

    impl Hash for Creature {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.tag.hash(state)
        }
    }

    impl Creature {}

    impl Genome for Creature {
        type Allele = Op;

        fn chromosome(&self) -> &[Self::Allele] {
            &self.chromosome
        }

        fn chromosome_mut(&mut self) -> &mut [Self::Allele] {
            &mut self.chromosome
        }

        fn random<H: Hash>(config: &Config, salt: H) -> Self
        where
            Self: Sized,
        {
            // First, let's get some information from the lifted program
            let mut rng = hash_seed_rng(&salt);
            let length = rng.gen_range(config.push_vm.min_len, config.push_vm.max_len);
            let ops = random_ops(&mut rng, config.push_vm.literal_rate, length);

            Self {
                chromosome: ops,
                chromosome_parentage: vec![],
                chromosome_mutation: vec![],
                tag: rng.gen::<u64>(),
                name: util::name::random(4, rng.gen::<u64>()),
                parents: vec![],
                generation: 0,
                payload: None,
                profile: None,
                fitness: None,
                front: None,
                num_offspring: 0,
                native_island: config.island_identifier,
                description: None,
            }
        }

        // is this simply identical to the original roper crossover?
        // this and much else could probably be defined generically.
        // TODO: Refactor
        fn crossover(mates: &[&Self], config: &Config) -> Self
        where
            Self: Sized,
        {
            let min_mate_len = mates.iter().map(|p| p.len()).min().unwrap();
            let lambda = min_mate_len as f64 / config.crossover_period;
            let distribution =
                rand_distr::Exp::new(lambda).expect("Failed to create random distribution");
            let parental_chromosomes = mates.iter().map(|m| m.chromosome()).collect::<Vec<_>>();
            let mut rng = hash_seed_rng(&mates[0]);
            let (chromosome, chromosome_parentage, parent_names) =
                // Check to see if we're performing a crossover or just cloning
                if rng.gen_range(0.0, 1.0) < config.crossover_rate {
                    let (c, p) = Self::crossover_by_distribution(&distribution, &parental_chromosomes);
                    let names = mates.iter().map(|p| p.name.clone()).collect::<Vec<String>>();
                    (c, p, names)
                } else {
                    let parent = parental_chromosomes[rng.gen_range(0, 2)];
                    let chromosome = parent.to_vec();
                    let parentage =
                        chromosome.iter().map(|_| 0).collect::<Vec<usize>>();
                    (chromosome, parentage, vec![mates[0].name.clone()])
                };

            let generation = mates.iter().map(|p| p.generation).max().unwrap() + 1;
            let name = util::name::random(4, &chromosome);
            let len = chromosome.len();

            Self {
                chromosome,
                chromosome_parentage,
                chromosome_mutation: vec![None; len],
                tag: rand::random::<u64>(),
                parents: parent_names,
                generation,
                name,
                profile: None,
                fitness: None,
                front: None,
                num_offspring: 0,
                native_island: config.island_identifier,
                description: None,
                ..Default::default()
            }
        }

        fn mutate(&mut self, config: &Config) {
            unimplemented!()
        }

        fn incr_num_offspring(&mut self, _n: usize) {
            unimplemented!()
        }
    }

    impl Phenome for Creature {
        type Fitness = Fitness<'static>;
        type Problem = ();

        fn fitness(&self) -> Option<&Self::Fitness> {
            self.fitness.as_ref()
        }

        fn scalar_fitness(&self) -> Option<f64> {
            self.fitness.as_ref().map(HasScalar::scalar)
        }

        fn priority_fitness(&self, config: &Config) -> Option<f64> {
            let priority = &config.fitness.priority;
            self.fitness().as_ref().and_then(|f| f.get(priority))
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

        fn store_answers(&mut self, results: Vec<Self::Problem>) {
            unimplemented!()
        }

        fn is_goal_reached(&self, config: &Config) -> bool {
            self.priority_fitness(config)
                .map(|p| p - config.fitness.target <= std::f64::EPSILON)
                .unwrap_or(false)
        }
    }

    // impl fmt::Debug for Creature<Op> {
    //     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    //         writeln!(f, "Name: {}, from island {}", self.name, self.native_island)?;
    //         writeln!(f, "Generation: {}", self.generation)?;
    //         let memory = loader::get_static_memory_image();
    //         for i in 0..self.chromosome.len() {
    //             let parent = if self.parents.is_empty() {
    //                 "seed"
    //             } else {
    //                 &self.parents[self.chromosome_parentage[i]]
    //             };
    //             let allele = &self.chromosome[i];
    //             let mutation = self.chromosome_mutation[i];
    //             writeln!(
    //                 f,
    //                 "[{}][{}] 0x{:010x}{}{} {}",
    //                 i,
    //                 parent,
    //                 allele,
    //                 perms,
    //                 if was_it_executed { " *" } else { "" },
    //                 mutation
    //                     .map(|m| format!("{:?}", m))
    //                     .unwrap_or_else(String::new),
    //             )?;
    //         }
    //         if let Some(ref profile) = self.profile {
    //             writeln!(f, "Trace:")?;
    //             for path in profile.disas_paths() {
    //                 writeln!(f, "{}", path)?;
    //             }
    //             //writeln!(f, "Register state: {:#x?}", profile.registers)?;
    //             for state in &profile.registers {
    //                 writeln!(f, "\nSpidered register state:\n{:?}", state)?;
    //             }
    //             writeln!(f, "CPU Error code(s): {:?}", profile.cpu_errors)?;
    //         }
    //         // writeln!(
    //         //     f,
    //         //     "Scalar fitness: {:?}",
    //         //     self.fitness().as_ref().map(|f| f.scalar())
    //         // )?;
    //         writeln!(f, "Fitness: {:#?}", self.fitness())?;
    //         Ok(())
    //     }
    //}
}

#[cfg(test)]
mod test {
    use unicorn::{Arch, Mode};

    use crate::configure::RoperConfig;

    use super::*;

    #[test]
    fn test_random_ops() {
        // first initialize things
        let mut config = RoperConfig {
            gadget_file: None,
            output_registers: vec![],
            randomize_registers: false,
            register_pattern: None,
            parsed_register_pattern: None,
            soup: None,
            soup_size: None,
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
        };
        crate::logger::init("test");
        println!("Loading, linking, and lifting...");
        loader::falcon_loader::load_from_path(&mut config, true);
        let ops = random_ops(&mut rand::thread_rng(), 0.5, 100);
        println!("{:#?}", ops);
        let mut machine = MachineState::default();
        let args = vec![];
        let res = machine.exec(&ops, &args, 1000);
        println!("Result: {:#x?}", res);
    }
}
