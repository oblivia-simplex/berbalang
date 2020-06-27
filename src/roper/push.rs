use hashbrown::HashMap;

use crate::configure::Config;
use crate::emulator::loader::get_static_memory_image;
use crate::util::architecture::{read_integer, write_integer, Perms};

pub type Stack<T> = Vec<T>;

pub type Input = Type;
pub type Output = Type;

// TODO: define a distribution over these ops. WordConst should comprise about half
// of any genome, I think, since without that, there's no ROP chain.

#[derive(Copy, Clone, Debug, Hash)]
pub enum Op {
    BoolAnd(Input, Input),
    BoolOr(Input, Input),
    BoolNot(Input),

    WordConst(u64),
    WordLess(Input, Input),
    WordEqual(Input, Input),
    WordAdd(Input, Input),
    WordSub(Input, Input),
    WordAnd(Input, Input),
    WordOr(Input, Input),
    WordXor(Input, Input),
    WordDiv(Input, Input),
    WordMult(Input, Input),
    WordOnes(Input),

    WordDeref(Input),
    WordSearch(Input),
    WordPerm(Input),
    WordToFloat(Input),

    WordReadable(Input),
    WordWriteable(Input),
    WordExecutable(Input),

    FloatLog(Input),
    FloatSin(Input),
    FloatCos(Input),
    FloatTan(Input),
    FloatTanh(Input),
    FloatToWord(Input),

    Rot(Type),
    Swap(Type),
    Drop(Type),
    Dup(Type),
}

impl Op {
    pub fn eval(&self, mach: &mut MachineState) {
        use Op::*;
        use Val::*;
        match self {
            // Generic Operations
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
                    mach.push(a);
                    mach.push(a);
                }
            }

            // Boolean Operations
            BoolAnd(a, b) => {
                if let (Bool(a), Bool(b)) = (mach.pop(a), mach.pop(b)) {
                    let res = Bool(a && b);
                    mach.push(res)
                }
            }
            BoolOr(a, b) => {
                if let (Bool(a), Bool(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Bool(a || b))
                }
            }
            BoolNot(a) => {
                if let Bool(a) = mach.pop(a) {
                    mach.push(Bool(!a))
                }
            }

            // Word (integer/address) Operations
            WordConst(w) => mach.push(Word(*w)),
            WordLess(a, b) => {
                if let (Word(a), Word(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Bool(a < b))
                }
            }
            WordEqual(a, b) => {
                if let (Word(a), Word(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Bool(a == b))
                }
            }
            WordAdd(a, b) => {
                if let (Word(a), Word(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Word(a.wrapping_add(b)))
                }
            }
            WordSub(a, b) => {
                if let (Word(a), Word(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Word(a.wrapping_sub(b)))
                }
            }
            WordAnd(a, b) => {
                if let (Word(a), Word(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Word(a & b))
                }
            }
            WordOr(a, b) => {
                if let (Word(a), Word(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Word(a | b))
                }
            }
            WordXor(a, b) => {
                if let (Word(a), Word(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Word(a ^ b))
                }
            }
            WordMult(a, b) => {
                if let (Word(a), Word(b)) = (mach.pop(a), mach.pop(b)) {
                    mach.push(Word(a.wrapping_mul(b)))
                }
            }
            WordOnes(a) => {
                if let Word(a) = mach.pop(a) {
                    mach.push(Word(a.count_ones() as u64))
                }
            }
            WordDeref(a) => {
                if let Word(a) = mach.pop(a) {
                    let memory = get_static_memory_image();
                    if let Some(x) = memory.try_dereference(a, None) {
                        if let Some(w) = read_integer(x, memory.endian, memory.word_size) {
                            mach.push(Word(w))
                        }
                    }
                }
            }
            WordSearch(a) => {
                if let Word(a) = mach.pop(a) {
                    let memory = get_static_memory_image();
                    let mut bytes = vec![0; memory.word_size];
                    write_integer(memory.endian, memory.word_size, a, &mut bytes);
                    if let Some(addr) = memory.seek_all_segs(&bytes, None) {
                        mach.push(Word(addr))
                    }
                }
            }
            WordReadable(a) => {
                if let Word(a) = mach.pop(a) {
                    let memory = get_static_memory_image();
                    let perm = memory.perm_of_addr(a).unwrap_or(Perms::NONE);
                    mach.push(Bool(perm.intersects(Perms::READ)))
                }
            }
            WordWriteable(a) => {
                if let Word(a) = mach.pop(a) {
                    let memory = get_static_memory_image();
                    let perm = memory.perm_of_addr(a).unwrap_or(Perms::NONE);
                    mach.push(Bool(perm.intersects(Perms::WRITE)))
                }
            }
            WordExecutable(a) => {
                if let Word(a) = mach.pop(a) {
                    let memory = get_static_memory_image();
                    let perm = memory.perm_of_addr(a).unwrap_or(Perms::NONE);
                    mach.push(Bool(perm.intersects(Perms::EXEC)))
                }
            }
            FloatLog(a) => {
                if let Float(a) = mach.pop(a) {
                    mach.push(Float(f64::from_bits(a).ln().to_bits()))
                }
            }
            FloatCos(a) => {
                if let Float(a) = mach.pop(a) {
                    mach.push(Float(f64::from_bits(a).cos().to_bits()))
                }
            }
            FloatSin(a) => {
                if let Float(a) = mach.pop(a) {
                    mach.push(Float(f64::from_bits(a).sin().to_bits()))
                }
            }
            FloatTanh(a) => {
                if let Float(a) = mach.pop(a) {
                    mach.push(Float(f64::from_bits(a).tanh().to_bits()))
                }
            }
            FloatTan(a) => {
                if let Float(a) = mach.pop(a) {
                    mach.push(Float(f64::from_bits(a).tan().to_bits()))
                }
            }

            FloatToWord(a) => {
                if let Float(a) = mach.pop(a) {
                    mach.push(Word(a))
                }
            }

            WordToFloat(a) => {
                if let Word(a) = mach.pop(a) {
                    mach.push(Float(a))
                }
            }

            _ => unimplemented!("no"),
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    Word,
    Byte,
    // Not using byte slices yet. but we could try later.
    Bool,
    Exec,
    Perm,
    None,
    Float,
}

impl From<Val> for Type {
    fn from(v: Val) -> Self {
        match v {
            Val::Word(_) => Type::Word,
            Val::Byte(_) => Type::Byte,
            Val::Bool(_) => Type::Bool,
            Val::Exec(_) => Type::Exec,
            Val::Float(_) => Type::Float,
            Val::Null => Type::None,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash)]
pub enum Val {
    Null,
    Word(u64),
    Byte(&'static [u8]),
    Bool(bool),
    Exec(Op),
    Float(u64),
}

impl Val {
    pub fn unwrap_word(self) -> Option<u64> {
        if let Self::Word(w) = self {
            Some(w)
        } else {
            None
        }
    }
}

pub struct MachineState(HashMap<Type, Stack<Val>>);

// TODO try optimizing by getting rid of the hashmap in favour of just
// using struct fields
impl MachineState {
    pub fn load_args(&mut self, args: &[Val]) {
        for arg in args {
            self.push(*arg)
        }
    }

    pub fn flush(&mut self) {
        self.0 = HashMap::new();
        self.0.insert(Type::Word, vec![]);
        self.0.insert(Type::Bool, vec![]);
        self.0.insert(Type::Float, vec![]);
        self.0.insert(Type::Exec, vec![]);
    }

    // the only reason for using Val::None is to make the code
    // a bit more ergonomic
    pub fn pop_opt(&mut self, t: &Type) -> Option<Val> {
        self.0.get_mut(t).expect("missing stack").pop()
    }

    pub fn pop(&mut self, t: &Type) -> Val {
        self.pop_opt(t).unwrap_or(Val::Null)
    }

    pub fn push(&mut self, val: Val) {
        let s = val.into();
        if let Type::None = s {
            return;
        }
        self.0.get_mut(&s).expect("missing output stack").push(val)
    }

    pub fn exec(&mut self, code: &[Op], args: &[Val], config: &Config) -> Vec<u64> {
        self.flush();
        self.load_args(args);
        // Load the exec stack
        for op in code {
            self.push(Val::Exec(*op))
        }

        let mut counter = config.push_vm.max_steps;

        while let Some(Val::Exec(op)) = self.pop_opt(&Type::Exec) {
            counter -= 1;
            if counter == 0 {
                break;
            }

            op.eval(self)
        }

        self.0
            .get_mut(&Type::Word)
            .unwrap()
            .drain(..)
            .filter_map(Val::unwrap_word)
            .collect()
    }
}

pub mod creature {}
