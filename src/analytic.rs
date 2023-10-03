use nalgebra::DMatrix;

use crate::FloatType;


pub fn cal_2nd_order_correlation(
    detection_events: &DMatrix<FloatType>,
    detector_mask: Option<&Vec<bool>>,
) -> (Vec<FloatType>, DMatrix<FloatType>) {
    if let Some(mask) = detector_mask {
        let indices: Vec<usize> = mask.iter().enumerate()
            .filter_map(|(i, &flag)| if !flag { Some(i) } else { None })
            .collect();
        analytical_core(&detection_events.select_columns(&indices))
    } else {
        analytical_core(detection_events)
    }
}

fn cal_two_points_expects(detection_events: &DMatrix<FloatType>) -> DMatrix<FloatType> {
    let num_shots = detection_events.nrows() as FloatType;
    let expect_ixj = detection_events.transpose() * detection_events;
    expect_ixj / num_shots
}

fn analytical_core(detection_events: &DMatrix<FloatType>) -> (Vec<FloatType>, DMatrix<FloatType>) {
    let num_dets = detection_events.ncols();
    let mut correlation_edges = DMatrix::<FloatType>::zeros(num_dets, num_dets);
    let mut correlation_bdy = Vec::<FloatType>::with_capacity(num_dets);
    let expect_ixj = cal_two_points_expects(detection_events);
    for i in 0..num_dets {
        let xi = expect_ixj[(i, i)];
        for j in 0..i {
            let xj = expect_ixj[(j, j)];
            let xij = expect_ixj[(i, j)];
            let denom = 1.0 - 2.0 * xi - 2.0 * xj + 4.0 * xij;
            let num = xij - xi * xj;
            let under_sqrt = 1.0 - 4.0 * num / denom;

            let pij = if under_sqrt > 0.0 {
                0.5 - 0.5 * under_sqrt.sqrt()
            } else {
                0.0
            };
            correlation_edges[(i, j)] = pij;
        }
    }

    correlation_edges += &correlation_edges.transpose();

    for i in 0..num_dets {
        let xi = expect_ixj[(i, i)];
        let pi_sum = correlation_edges.row(i).iter().fold(0.0, |p, &q| p + q - 2.0 * p * q);
        let pi_bdy = (xi - pi_sum) / (1.0 - 2.0 * pi_sum);
        correlation_bdy.push(pi_bdy);
    }

    (correlation_bdy, correlation_edges)
}