use std::fmt::Debug;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use deflate::write::GzEncoder;
use deflate::Compression;
use serde::{de::DeserializeOwned, Serialize};

use crate::error::Error;

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

pub fn undump<T: DeserializeOwned, P: AsRef<Path> + Debug>(path: P) -> Result<T, Error> {
    let mut file = fs::File::open(&path)?;
    let mut bytes = Vec::new();
    let num_bytes = file.read_to_end(&mut bytes)?;
    log::debug!("{} bytes read from {:?}", num_bytes, path);
    let decompressed: Vec<u8> = deflate::deflate_bytes_gzip(&bytes);
    let thing = serde_json::de::from_reader::<&[u8], T>(&decompressed)?;
    Ok(thing)
}

pub fn ron_dump<T: Serialize, P: AsRef<Path> + Debug>(thing: T, path: P) -> Result<(), Error> {
    let mut file = fs::File::create(&path)?;
    let mut dumper = || -> Result<(), Error> {
        let mut gz = GzEncoder::new(Vec::new(), Compression::Default);
        let s = ron::ser::to_string(&thing)?;
        gz.write_all(s.as_bytes());
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

pub fn ron_undump<T: DeserializeOwned, P: AsRef<Path> + Debug>(path: P) -> Result<T, Error> {
    let mut file = fs::File::open(&path)?;
    let mut bytes = Vec::new();
    let num_bytes = file.read_to_end(&mut bytes)?;
    log::debug!("{} bytes read from {:?}", num_bytes, path);
    let decompressed: Vec<u8> = deflate::deflate_bytes_gzip(&bytes);
    let thing = ron::de::from_reader::<&[u8], T>(&decompressed)?;
    Ok(thing)
}
