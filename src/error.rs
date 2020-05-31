use std::io;

#[derive(Debug)]
pub enum Error {
    IO(io::Error),
    ParseInt(std::num::ParseIntError),
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
