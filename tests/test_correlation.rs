use correlation::{cal_2nd_order_correlation, read_b8_file, cal_high_order_correlations};
use std::time::Instant;

#[test]
fn test_2nd_correlation() {
    let metadata = std::fs::File::open("test_data/surface_code/metadata.yaml").unwrap();
    let metadata: serde_yaml::Value = serde_yaml::from_reader(metadata).unwrap();
    let num_detectors = metadata["num_detectors"].as_u64().unwrap() as usize;

    let dets = read_b8_file("test_data/surface_code/detectors.b8", num_detectors).unwrap();
    let start = Instant::now();
    let (bdy, edges) = cal_2nd_order_correlation(&dets, None);
    let duration = start.elapsed();
    println!("Time elapsed in cal_2nd_order_correlation() is: {:?}", duration);
    assert_eq!(bdy.len(), num_detectors);
    assert_eq!(edges.shape(), (num_detectors, num_detectors));
    // println!("{:#?}", bdy);
}

#[test]
fn test_high_order_correlation_rep_code() {
    let metadata = std::fs::File::open("test_data/rep_code/metadata.yaml").unwrap();
    let metadata: serde_yaml::Value = serde_yaml::from_reader(metadata).unwrap();
    let num_detectors = metadata["num_detectors"].as_u64().unwrap() as usize;

    let dets = read_b8_file("test_data/rep_code/detectors.b8", num_detectors).unwrap();
    let start = Instant::now();
    // let (bdy, edges) = cal_2nd_order_correlation(&dets, None);
    let res = cal_high_order_correlations(&dets, None, 1);
    let duration = start.elapsed();
}
