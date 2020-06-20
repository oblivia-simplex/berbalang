use std::fs;
use std::io::Write;
use std::path::Path;

use deflate::write::GzEncoder;
use deflate::Compression;
use serde::Serialize;

use crate::error::Error;
use std::fmt::Debug;

pub fn dump<T: Serialize, P: AsRef<Path> + Debug>(thing: T, path: P) -> Result<(), Error> {
    let mut file = fs::File::create(&path)?;
    let mut dumper = || -> Result<(), Error> {
        let mut gz = GzEncoder::new(Vec::new(), Compression::Default);
        serde_json::to_writer(&mut gz, &thing)?;
        let compressed = gz.finish()?;
        file.write_all(&compressed).map_err(Error::from)
    };
    if let Err(e) = dumper() {
        fs::remove_file(&path)?;
        log::warn!("Failed to dump to {:?}, removed file", path);
        Err(e)
    } else {
        Ok(())
    }
}
