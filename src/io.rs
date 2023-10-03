use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::Result;
use bitvec::prelude::*;
use nalgebra::DMatrix;

use crate::FloatType;


pub fn read_b8_file<P>(
    path: P,
    num_bits_per_shot: usize,
) -> Result<DMatrix<FloatType>>
    where
        P: AsRef<Path>,
{
    let data = std::fs::read(path)?;
    let num_bytes_per_shot = (num_bits_per_shot + 7) / 8;
    let num_shots = data.len() / num_bytes_per_shot;
    let bits_iter = data
        .chunks(num_bytes_per_shot)
        .flat_map(|chunk| {
            let mut bv = BitVec::<_, Lsb0>::from_slice(chunk);
            bv.truncate(num_bits_per_shot);
            bv.into_iter().map(|b| b as u8 as FloatType)
        });
    let detection_events = DMatrix::from_row_iterator(
        num_shots,
        num_bits_per_shot,
        bits_iter,
    );
    Ok(detection_events)
}

pub fn read_01_file<P>(
    path: P,
) -> Result<DMatrix<FloatType>>
    where
        P: AsRef<Path>
{
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    let lines = buf_reader.lines().collect::<Result<Vec<_>, _>>()?;
    let num_shots = lines.len();
    let num_bits_per_shot = lines[0].len();
    let bits_iter = lines
        .iter()
        .flat_map(|line| {
            line.chars().map(|c| if c == '1' { 1.0 as FloatType } else { 0.0 as FloatType })
        });
    let detection_events = DMatrix::from_row_iterator(
        num_shots,
        num_bits_per_shot,
        bits_iter,
    );
    Ok(detection_events)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_b8_file() {
        let path = "test_data/detectors.b8";
        let data = read_b8_file(path, 24).unwrap();
        assert_eq!(data.shape(), (1000, 24));
        assert_eq!(data[(0, 0)], 0.);
        assert_eq!(data[(0, 5)], 1.);
        assert_eq!(data[(4, 7)], 1.);
    }

    #[test]
    fn test_read_01_file() {
        let path = "test_data/detectors.01";
        let data = read_01_file(path).unwrap();
        assert_eq!(data.shape(), (1000, 24));
        assert_eq!(data[(0, 0)], 0.);
        assert_eq!(data[(0, 5)], 1.);
        assert_eq!(data[(4, 7)], 1.);
    }
}