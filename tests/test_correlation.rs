use correlation::{cal_2nd_order_correlation, cal_high_order_correlations, read_b8_file};
use smallvec::smallvec;

#[test]
fn test_2nd_correlation() {
    let metadata = std::fs::File::open("test_data/surface_code/metadata.yaml").unwrap();
    let metadata: serde_yaml::Value = serde_yaml::from_reader(metadata).unwrap();
    let num_detectors = metadata["num_detectors"].as_u64().unwrap() as usize;

    let dets = read_b8_file("test_data/surface_code/detectors.b8", num_detectors).unwrap();
    let (bdy, edges) = cal_2nd_order_correlation(&dets, None);
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
    let (bdy, edges) = cal_2nd_order_correlation(&dets, None);
    // println!("{}", bdy[0]);
    println!("{}", edges.index((2, 3)));
    let res = cal_high_order_correlations(&dets, None, 16).unwrap();
    let bdy_high: Vec<_> = (0..num_detectors).map(|i| res[&smallvec![i]]).collect();
    // println!("{bdy}");
    // println!("{:#?}", bdy_high);
    // println!("{:#?}", res);
    // println!("{}", bdy_high[1])
    println!("{}", res[&smallvec![2, 3]])
}
