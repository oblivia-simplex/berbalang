use std::fmt::Debug;
use std::ops::{Add, Div};

use serde::Serialize;

pub trait FitnessScoreReq:
    PartialEq + Debug + Send + Clone + PartialOrd + Serialize + Add + Div
{
}

pub trait FitnessScore:
    PartialEq + Debug + Send + Clone + PartialOrd + Serialize + Ord
{
}

impl FitnessScore for usize {}
impl FitnessScore for (usize, usize) {}
