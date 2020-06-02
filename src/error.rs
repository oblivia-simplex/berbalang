use std::fmt::Debug;
use std::io;

#[derive(Debug)]
pub enum Error {
    IO(io::Error),
    ParseInt(std::num::ParseIntError),
    Channel(String),
}

macro_rules! impl_error_from {
    ($e:ty, $var:ident) => {
        impl From<$e> for Error {
            fn from(e: $e) -> Self {
                Self::$var(e)
            }
        }
    };
}

impl_error_from!(std::io::Error, IO);
impl_error_from!(std::num::ParseIntError, ParseInt);

impl<T: Debug> From<std::sync::mpsc::SendError<T>> for Error {
    fn from(e: std::sync::mpsc::SendError<T>) -> Self {
        Self::Channel(format!("{:?}", e))
    }
}

impl From<std::sync::mpsc::RecvError> for Error {
    fn from(e: std::sync::mpsc::RecvError) -> Self {
        Self::Channel(format!("{:?}", e))
    }
}
