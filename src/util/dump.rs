use std::fs;
use std::io::Write;
use std::path::Path;

use deflate::write::GzEncoder;
use deflate::Compression;
use serde::Serialize;

use crate::error::Error;

pub fn dump<T: Serialize, P: AsRef<Path>>(thing: T, path: P) -> Result<(), Error> {
    let mut file = fs::File::create(&path)?;
    let mut gz = GzEncoder::new(Vec::new(), Compression::Default);
    serde_json::to_writer(&mut gz, &thing)?;
    let compressed = gz.finish()?;
    file.write_all(&compressed)?;
    Ok(())
}
