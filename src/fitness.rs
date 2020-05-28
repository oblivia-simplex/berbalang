use std::fmt::Debug;
use std::ops::{Add, Div};

use serde::Serialize;

pub trait FitnessScoreReq:
    PartialEq + Debug + Send + Clone + PartialOrd + Serialize + Add + Div
{
}

pub trait FitnessScore:
    PartialEq + Debug + Send + Clone + PartialOrd + Serialize + PartialOrd
{
}

// this smells bad
impl FitnessScore for usize {}
impl FitnessScore for (usize, usize) {}
impl FitnessScore for (usize, f32) {}
impl FitnessScore for (usize, f32, usize) {}
impl FitnessScore for (usize, f32, f32, usize) {}
impl FitnessScore for Vec<f32> {}
