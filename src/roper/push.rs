use falcon::il;
use hashbrown::HashMap;

use crate::configure::Config;
use crate::emulator::loader::get_static_memory_image;
use crate::util::architecture::{read_integer, write_integer, Perms};

pub type Stack<T> = Vec<T>;

pub type Input = Type;
pub type Output = Type;

// TODO: define a distribution over these ops. IntConst should comprise about half
// of any genome, I think, since without that, there's no ROP chain.

/// Virtual
/// Instruction
/// Network
/// Domain
/// Specific
/// Language
///
/// VINDSL
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Op {
    BoolAnd,
    BoolOr,
    BoolNot,

    IntConst(u64),
    IntLess,
    IntEqual,
    IntAdd,
    IntSub,
    IntAnd,
    IntOr,
    IntXor,
    IntDiv,
    IntMul,
    IntMod,
    IntOnes,

    IntDeref,
    IntSearch,
    IntToFloat,

    IntReadable,
    IntWriteable,
    IntExecutable,

    IntToGadget,
    GadgetToInt,

    // floats need to be encoded as u64 to preserve Hash
    FloatConst(u64),
    FloatLog,
    FloatSin,
    FloatCos,
    FloatTan,
    FloatTanh,
    FloatToInt,

    CodeQuote,
    CodeDoAll,

    // Falcon Instructions
    InstIsBranch,
    InstIsStore,
    InstIsLoad,
    InstAddr,

    BlockInsts,
    BlockAddr,
    BlockSuccs,
    BlockPreds,

    FuncBlock,
    FuncEntry,
    FuncExit,
    FuncAddr,

    ExprToInt,
    // for constants
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

            // Int (integer/address) Operations
            IntConst(w) => mach.push(Int(*w)),
            IntLess => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    mach.push(Bool(a < b))
                }
            }
            IntEqual => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    mach.push(Bool(a == b))
                }
            }
            IntAdd => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    mach.push(Int(a.wrapping_add(b)))
                }
            }
            IntSub => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    mach.push(Int(a.wrapping_sub(b)))
                }
            }
            IntAnd => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    mach.push(Int(a & b))
                }
            }
            IntOr => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    mach.push(Int(a | b))
                }
            }
            IntXor => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    mach.push(Int(a ^ b))
                }
            }
            IntMul => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    mach.push(Int(a.wrapping_mul(b)))
                }
            }
            IntDiv => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    if b != 0 {
                        mach.push(Int(a / b))
                    }
                }
            }
            IntMod => {
                if let (Int(a), Int(b)) = (mach.pop(&Type::Int), mach.pop(&Type::Int)) {
                    if b != 0 {
                        mach.push(Int(a % b))
                    }
                }
            }
            IntOnes => {
                if let Int(a) = mach.pop(&Type::Int) {
                    mach.push(Int(a.count_ones() as u64))
                }
            }
            IntDeref => {
                if let Int(a) = mach.pop(&Type::Int) {
                    let memory = get_static_memory_image();
                    if let Some(x) = memory.try_dereference(a, None) {
                        if let Some(w) = read_integer(x, memory.endian, memory.word_size) {
                            mach.push(Int(w))
                        }
                    }
                }
            }
            IntSearch => {
                if let Int(a) = mach.pop(&Type::Int) {
                    let memory = get_static_memory_image();
                    let mut bytes = vec![0; memory.word_size];
                    write_integer(memory.endian, memory.word_size, a, &mut bytes);
                    if let Some(addr) = memory.seek_all_segs(&bytes, None) {
                        mach.push(Int(addr))
                    }
                }
            }
            IntReadable => {
                if let Int(a) = mach.pop(&Type::Int) {
                    let memory = get_static_memory_image();
                    let perm = memory.perm_of_addr(a).unwrap_or(Perms::NONE);
                    mach.push(Bool(perm.intersects(Perms::READ)))
                }
            }
            IntWriteable => {
                if let Int(a) = mach.pop(&Type::Int) {
                    let memory = get_static_memory_image();
                    let perm = memory.perm_of_addr(a).unwrap_or(Perms::NONE);
                    mach.push(Bool(perm.intersects(Perms::WRITE)))
                }
            }
            IntExecutable => {
                if let Int(a) = mach.pop(&Type::Int) {
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

            FloatToInt => {
                if let Float(a) = mach.pop(&Type::Float) {
                    mach.push(Int(a))
                }
            }

            IntToFloat => {
                if let Int(a) = mach.pop(&Type::Int) {
                    mach.push(Float(a))
                }
            }
            IntToGadget => {
                if let Int(a) = mach.pop(&Type::Int) {
                    mach.push(Gadget(a))
                }
            }
            GadgetToInt => {
                if let Gadget(a) = mach.pop(&Type::Gadget) {
                    mach.push(Int(a))
                }
            }

            ExprToInt => {
                if let Expression(il::Expression::Constant(c)) = mach.pop(&Type::Expression) {
                    if let Some(n) = c.value_u64() {
                        mach.push(Int(n))
                    }
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
                            mach.push(Int(n))
                        }
                    } else {
                        let (op, a, b) = match expr {
                            E::Add(a, b) => (IntAdd, Some(a), Some(b)),
                            E::Sub(a, b) => (IntSub, Some(a), Some(b)),
                            E::Mul(a, b) => (IntMul, Some(a), Some(b)),
                            E::Divu(a, b) => (IntDiv, Some(a), Some(b)),
                            E::Modu(a, b) => (IntMod, Some(a), Some(b)),
                            E::Divs(a, b) => (IntDiv, Some(a), Some(b)),
                            E::Mods(a, b) => (IntMod, Some(a), Some(b)),
                            E::And(a, b) => (IntAnd, Some(a), Some(b)),
                            E::Or(a, b) => (IntOr, Some(a), Some(b)),
                            E::Xor(a, b) => (IntXor, Some(a), Some(b)),
                            E::Cmpeq(a, b) => (Eq(Type::Int), Some(a), Some(b)),
                            E::Cmpneq(a, b) => (Eq(Type::Int), Some(a), Some(b)),
                            E::Cmplts(a, b) => (IntLess, Some(a), Some(b)),
                            E::Cmpltu(a, b) => (IntLess, Some(a), Some(b)),
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
                if let (Function(f), Int(i)) = (mach.pop(&Type::Function), mach.pop(&Type::Int)) {
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
                    mach.push(Int(addr));
                }
            }

            BlockAddr => {
                if let Block(a) = mach.pop(&Type::Block) {
                    if let Some(addr) = a.address() {
                        mach.push(Int(addr));
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
                        mach.push(Int(addr))
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    Int,
    Byte,
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
    Operation,
    Instruction,
    Block,
    Function,
}

impl From<&Val> for Type {
    fn from(v: &Val) -> Self {
        match v {
            Val::Int(_) => Type::Int,
            Val::Gadget(_) => Type::Gadget,
            Val::Byte(_) => Type::Byte,
            Val::Bool(_) => Type::Bool,
            Val::Exec(_) => Type::Exec,
            Val::Float(_) => Type::Float,
            Val::Code(_) => Type::Code,
            Val::Scalar(_) => Type::Scalar,
            Val::Expression(_) => Type::Expression,
            Val::Operation(_) => Type::Operation,
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
    Int(u64),
    Byte(&'static [u8]),
    Bool(bool),
    Exec(Op),
    Code(Op),
    Float(u64),
    Gadget(u64),
    // Falcon
    Scalar(il::Scalar),
    Expression(il::Expression),
    Operation(&'static il::Operation),
    Instruction(&'static il::Instruction),
    Block(&'static il::Block),
    Function(&'static il::Function),
}

impl Val {
    pub fn unwrap_word(self) -> Option<u64> {
        match self {
            Self::Int(w) | Self::Gadget(w) => Some(w),
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
        self.stacks.insert(Type::Int, vec![]);
        self.stacks.insert(Type::Bool, vec![]);
        self.stacks.insert(Type::Float, vec![]);
        self.stacks.insert(Type::Exec, vec![]);
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

    pub fn exec(&mut self, code: &[Op], args: &[Val], config: &Config) -> Vec<u64> {
        self.flush();
        self.load_args(args);
        // Load the exec stack
        for op in code {
            self.push(Val::Exec(op.clone()))
        }

        while let Some(Val::Exec(op)) = self.pop_opt(&Type::Exec) {
            self.counter += 1;
            if self.counter >= config.push_vm.max_steps {
                break;
            }

            op.eval(self)
        }

        // first, take the explicitly-marked gadgets.
        // ensure that this list begins with an executable.
        let mut gadgets: Vec<u64> = self
            .stacks
            .get_mut(&Type::Gadget)
            .unwrap()
            .drain(..)
            .filter_map(Val::unwrap_word)
            .collect();
        let memory = get_static_memory_image();
        while !gadgets
            .last()
            .and_then(|a| memory.perm_of_addr(*a))
            .map(|p| p.intersects(Perms::EXEC))
            .unwrap_or(false)
        {
            let _ = gadgets.pop();
        }
        gadgets.reverse();

        let integers = self
            .stacks
            .get_mut(&Type::Int)
            .unwrap()
            .drain(..)
            .filter_map(Val::unwrap_word);
        gadgets.extend(integers);
        gadgets
    }
}

pub mod creature {}
