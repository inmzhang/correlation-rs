use correlation::{
    cal_2nd_order_correlation, cal_high_order_correlations, read_b8_file, HyperEdge,
};
use itertools::Itertools;
use smallvec::smallvec;
use std::collections::HashSet;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct HyperedgeSetup {
    hyperedges: Vec<HashSet<usize>>,
    probability: Vec<f64>,
}

fn run() {
    let metadata = std::fs::File::open("test_data/surface_code/metadata.yaml").unwrap();
    let metadata: serde_yaml::Value = serde_yaml::from_reader(metadata).unwrap();
    let num_detectors = metadata["num_detectors"].as_u64().unwrap() as usize;
    let dets = read_b8_file("test_data/surface_code/detectors.b8", num_detectors).unwrap();

    // run analytically
    let start = Instant::now();
    let (bdy, edges) = cal_2nd_order_correlation(&dets, None);
    println!("Analytical time: {:?}us", start.elapsed().as_micros());
    println!(
        "First five elements of analytical boundary: {:?}",
        &bdy.iter().take(5).collect::<Vec<_>>()
    );
    println!(
        "First five elements of analytical edges: {:?}",
        &edges.row(0).iter().skip(1).take(5).collect::<Vec<_>>()
    );

    // run numerically but ignore hyperedges
    let start = Instant::now();
    let res = cal_high_order_correlations(&dets, None, None, 1e-6, 100).unwrap();
    println!("Numerical time: {:?}ms", start.elapsed().as_millis());
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

    // run numerically with hyperedges included
    let file = std::fs::File::open("test_data/surface_code/hyperedges.json").unwrap();
    let hyperedges_setup: HyperedgeSetup = serde_json::from_reader(file).unwrap();
    let start = Instant::now();
    let res =
        cal_high_order_correlations(&dets, Some(&hyperedges_setup.hyperedges), None, 1e-6, 100)
            .unwrap();
    println!(
        "Numerical time with hyperedges: {:?}s",
        start.elapsed().as_secs()
    );
    let show_hyperedges = hyperedges_setup
        .hyperedges
        .into_iter()
        .zip(hyperedges_setup.probability)
        .filter(|(h, _)| h.len() > 2)
        .collect_vec();
    println!(
        "Solved hyperedge probabilities: {:?}",
        show_hyperedges
            .iter()
            .map(|(h, _)| res[&h.iter().copied().sorted().collect::<HyperEdge>()])
            .collect_vec()
    );
    println!(
        "Analytical hyperedge probabilities: {:?}",
        show_hyperedges.iter().map(|(_, p)| p).collect_vec()
    );
}

fn main() {
    run()
}
