use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::Result;
use bitvec::prelude::*;
use itertools::Itertools;
use ndarray::Array2;

pub fn read_b8_file<P>(path: P, num_bits_per_shot: usize) -> Result<Array2<f64>>
where
    P: AsRef<Path>,
{
    let data = std::fs::read(path)?;
    let num_bytes_per_shot = (num_bits_per_shot + 7) / 8;
    let num_shots = data.len() / num_bytes_per_shot;
    let data_vec = data
        .chunks(num_bytes_per_shot)
        .flat_map(|chunk| {
            let mut bv = BitVec::<_, Lsb0>::from_slice(chunk);
            bv.truncate(num_bits_per_shot);
            bv.into_iter().map(|b| b as u8 as f64)
        })
        .collect_vec();
    Ok(Array2::from_shape_vec(
        (num_shots, num_bits_per_shot),
        data_vec,
    )?)
}

pub fn read_01_file<P>(path: P) -> Result<Array2<f64>>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    let lines = buf_reader.lines().collect::<Result<Vec<_>, _>>()?;
    let num_shots = lines.len();
    let num_bits_per_shot = lines[0].len();
    let data_vec = lines
        .iter()
        .flat_map(|line| {
            line.chars()
                .map(|c| if c == '1' { 1.0_f64 } else { 0.0_f64 })
        })
        .collect_vec();
    Ok(Array2::from_shape_vec(
        (num_shots, num_bits_per_shot),
        data_vec,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_file() {
        let metadata = std::fs::File::open("test_data/surface_code/metadata.yaml").unwrap();
        let metadata: serde_yaml::Value = serde_yaml::from_reader(metadata).unwrap();
        let num_detectors = metadata["num_detectors"].as_u64().unwrap() as usize;
        let num_shots = metadata["num_shots"].as_u64().unwrap() as usize;

        let path = "test_data/surface_code/detectors.b8";
        let data = read_b8_file(path, num_detectors).unwrap();
        assert_eq!(data.shape(), [num_shots, num_detectors]);
    }

    #[test]
    fn test_read_01_file() {
        let metadata = std::fs::File::open("test_data/surface_code/metadata.yaml").unwrap();
        let metadata: serde_yaml::Value = serde_yaml::from_reader(metadata).unwrap();
        let num_detectors = metadata["num_detectors"].as_u64().unwrap() as usize;
        let num_shots = metadata["num_shots"].as_u64().unwrap() as usize;

        let path = "test_data/surface_code/detectors.01";
        let data = read_01_file(path).unwrap();
        assert_eq!(data.shape(), [num_shots, num_detectors]);
    }
}
