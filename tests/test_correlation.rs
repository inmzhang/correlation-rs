use correlation::{cal_2nd_order_correlation, read_b8_file};
use std::time::Instant;

#[test]
fn test_2nd_correlation() {
    let dets = read_b8_file("test_data/detectors.b8", 4400).unwrap();
    let start = Instant::now();
    let (bdy, edges) = cal_2nd_order_correlation(&dets, None);
    let duration = start.elapsed();
    println!("Time elapsed in cal_2nd_order_correlation() is: {:?}", duration);
    assert_eq!(bdy.len(), 4400);
    assert_eq!(edges.shape(), (4400, 4400));
    // println!("{:#?}", bdy);
}
