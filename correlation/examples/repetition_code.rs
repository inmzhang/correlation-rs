use correlation::{cal_2nd_order_correlation, cal_high_order_correlations, read_b8_file};
use smallvec::smallvec;

fn run() {
    let metadata = std::fs::File::open("correlation/test_data/rep_code/metadata.yaml").unwrap();
    let metadata: serde_yaml::Value = serde_yaml::from_reader(metadata).unwrap();
    let num_detectors = metadata["num_detectors"].as_u64().unwrap() as usize;
    let dets = read_b8_file("correlation/test_data/rep_code/detectors.b8", num_detectors).unwrap();
    // run analytically
    let (bdy, edges) = cal_2nd_order_correlation(&dets);
    println!(
        "First five elements of analytical boundary: {:?}",
        &bdy.iter().take(5).collect::<Vec<_>>()
    );
    println!(
        "First five elements of analytical edges: {:?}",
        &edges.row(0).iter().skip(1).take(5).collect::<Vec<_>>()
    );
    // run numerically
    let res = cal_high_order_correlations(&dets, None, None, None).unwrap();
    let bdy_numeric = (0..num_detectors)
        .map(|i| res[&smallvec![i]])
        .collect::<Vec<_>>();
    let edges_first_five = [
        res[&smallvec![0, 1]],
        res[&smallvec![0, 2]],
        res[&smallvec![0, 3]],
        res[&smallvec![0, 4]],
        res[&smallvec![0, 5]],
    ];
    println!(
        "First five elements of numerical boundary: {:?}",
        &bdy_numeric.iter().take(5).collect::<Vec<_>>()
    );
    println!(
        "First five elements of numerical edges: {:?}",
        edges_first_five
    );
}

fn main() {
    run()
}
