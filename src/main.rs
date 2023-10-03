mod io;
mod analytic;

pub(crate) type FloatType = f32;

fn main() {
    let dets = io::read_b8_file("test_data/detectors.b8", 4400).unwrap();
    let (bdy, edges) = analytic::cal_2nd_order_correlation(&dets, None);
    println!("{}", bdy.len());
    println!("{:#?}", bdy);
    // println!("{:?}", edges);
}
