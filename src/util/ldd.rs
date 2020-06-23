use crate::error::Error;
use std::path::PathBuf;
use std::process::Command;

pub fn ldd(path: &str) -> Result<Vec<String>, Error> {
    let output = Command::new("ldd").arg(path).output()?.stdout;
    Ok(std::str::from_utf8(&output)?
        .lines()
        .filter_map(|s| s.split(" => ").nth(1))
        .filter_map(|s: &str| s.split(" ").nth(0))
        .map(String::from)
        .collect::<Vec<String>>())
}

pub fn ld_paths(path: &str) -> Result<Vec<String>, Error> {
    let ldd_out = ldd(path)?;
    let dirs = ldd_out
        .into_iter()
        .map(PathBuf::from)
        .map(|mut p| {
            let _f = p.pop();
            p
        })
        .map(|s| s.to_string_lossy().into())
        .collect::<Vec<String>>();
    Ok(dirs)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ldd() {
        let res = ldd("/bin/sh").expect("Failed");
        println!("ldd /bin/sh:");
        println!("{:#?}", res);
    }

    #[test]
    fn test_ld_paths() {
        let res = ld_paths("/bin/sh").expect("Failed");
        println!("ld_paths of /bin/sh");
        println!("{:#?}", res);
    }
}
