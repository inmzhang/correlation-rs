use std::collections::{HashMap, HashSet};

use anyhow::Error;
use gomez::nalgebra as na;
use gomez::nalgebra::{IsContiguous, Storage, StorageMut, Vector};
use gomez::prelude::*;
use gomez::solver::TrustRegion;
use itertools::Itertools;
use na::{DMatrix, DVector, Dynamic};
use rayon::prelude::*;
use smallvec::SmallVec;

pub use io::{read_01_file, read_b8_file};

mod io;

/// Calculate the 2nd order correlation analytically.
///
/// # Arguments
///
/// * `detection_events` - A matrix of detection events with shape (num_shots, num_detectors).
///
/// * `detector_mask` - Boolean mask to mask certain detectors out. If the i-th element is true,
/// the i-th detector is masked out.
///
/// # Returns
///
/// A tuple of two elements. The first element is a vector of boundary correlations. The second
/// element is a matrix of edge correlations.
pub fn cal_2nd_order_correlation(
    detection_events: &DMatrix<f64>,
    detector_mask: Option<&DVector<bool>>,
) -> (DVector<f64>, DMatrix<f64>) {
    if let Some(mask) = detector_mask {
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &flag)| if !flag { Some(i) } else { None })
            .collect();
        analytical_core(&detection_events.select_columns(&indices))
    } else {
        analytical_core(detection_events)
    }
}

fn cal_two_points_expects(detection_events: &DMatrix<f64>) -> DMatrix<f64> {
    let num_shots = detection_events.nrows() as f64;
    let expect_ixj = detection_events.transpose() * detection_events;
    expect_ixj / num_shots
}

fn analytical_core(detection_events: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let num_dets = detection_events.ncols();
    let mut correlation_edges = DMatrix::<f64>::zeros(num_dets, num_dets);
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

    let correlation_bdy = DVector::from_iterator(
        num_dets,
        (0..num_dets).map(|i| {
            let xi = expect_ixj[(i, i)];
            let pi_sum = correlation_edges
                .row(i)
                .iter()
                .fold(0.0, |p, &q| p + q - 2.0 * p * q);
            (xi - pi_sum) / (1.0 - 2.0 * pi_sum)
        }),
    );
    (correlation_bdy, correlation_edges)
}

pub type HyperEdge = SmallVec<[usize; 6]>;

type Cluster = Vec<usize>;

/// Calculate the high order correlations.
///
/// # Arguments
///
/// * `detection_events` - A matrix of detection events with shape (num_shots, num_detectors).
///
/// * `hyperedges` - A list of hyperedges. Each hyperedge is a set of detectors.
///
/// * `num_threads` - Number of threads to use in parallel, default to number of cpus.
///
/// * `max_iters` - Number of iterations the optimizer can take.
///
/// # Returns
///
/// A map from hyperedges to their correlation probabilities.
pub fn cal_high_order_correlations(
    detection_events: &DMatrix<f64>,
    hyperedges: Option<&[HashSet<usize>]>,
    num_threads: Option<usize>,
    atol: f64,
    max_iters: usize,
) -> Result<HashMap<HyperEdge, f64>, Error> {
    let num_detectors = detection_events.ncols();
    let all_hyperedges = all_hyperedges_considered(num_detectors, hyperedges);
    // divide the hyperedges into clusters
    let (extended_hyperedges, clusters) = cluster_hyperedges(&all_hyperedges);
    // calculate the expectations of each hyperedge
    let expectations = calculate_expectations(detection_events, &extended_hyperedges);
    // thread pool to use
    let pool = create_thread_pool(num_threads.unwrap_or(num_cpus::get()));
    // solve each cluster in parallel
    // let solved_probs = clusters
    //     .iter()
    //     .map(|cluster| solve_cluster(cluster, &extended_hyperedges, &expectations, max_iters))
    //     .collect::<Result<Vec<_>, Error>>()?;
    let solved_probs = pool.install(|| {
        clusters
            .par_iter()
            .map(|cluster| {
                solve_cluster(
                    cluster,
                    &extended_hyperedges,
                    &expectations,
                    atol,
                    max_iters,
                )
            })
            .collect::<Result<Vec<_>, Error>>()
    })?;
    // adjust probabilities
    let mut adjusted_probs = adjust_probabilities(&clusters, &extended_hyperedges, &solved_probs);
    // retain concerned hyperedges
    adjusted_probs.retain(|h, _| all_hyperedges.contains(h));
    Ok(adjusted_probs)
}

fn all_hyperedges_considered(
    num_detectors: usize,
    hyperedges: Option<&[HashSet<usize>]>,
) -> Vec<HyperEdge> {
    let mut all_hyperedges: Vec<HyperEdge> = (0..num_detectors)
        .map(|e| HyperEdge::from_iter(std::iter::once(e)))
        .collect_vec();
    all_hyperedges.extend(
        (0..num_detectors)
            .combinations(2)
            .map(|e| e.into_iter().collect()),
    );
    if let Some(hyperedges) = hyperedges {
        all_hyperedges.extend(hyperedges.iter().map(|e| {
            let mut hyperedge = HyperEdge::from_iter(e.clone());
            hyperedge.sort();
            hyperedge
        }));
    };
    all_hyperedges.sort_by_key(|h| h.len());
    all_hyperedges
}

fn create_thread_pool(num_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap()
}

fn cluster_hyperedges(hyperedges: &Vec<HyperEdge>) -> (Vec<HyperEdge>, Vec<Cluster>) {
    let mut extended_hyperedges = hyperedges.clone();
    let mut clusters = Vec::new();
    let mut indices_wait_cluster = (0..hyperedges.len()).collect_vec();
    while let Some(i) = indices_wait_cluster.pop() {
        let root = &hyperedges[i];
        let cluster = (1..=root.len())
            .flat_map(move |k| {
                root.clone()
                    .into_iter()
                    .combinations(k)
                    .map(|c| c.into_iter().collect::<HyperEdge>())
            })
            .map(|h| {
                if let Some(idx) = extended_hyperedges.iter().position(|e| *e == h) {
                    idx
                } else {
                    extended_hyperedges.push(h);
                    extended_hyperedges.len() - 1
                }
            })
            .collect_vec();
        indices_wait_cluster.retain(|j| !cluster.contains(j));
        clusters.push(cluster);
    }
    (extended_hyperedges, clusters)
}

fn calculate_expectations(
    detection_events: &DMatrix<f64>,
    extended_hyperedges: &[HyperEdge],
) -> Vec<f64> {
    let (num_shots, num_detectors) = detection_events.shape();
    let n_low_order = (num_detectors * (num_detectors + 1)) / 2;
    // pre-calculate 2-point expectations using
    // matrix multiply to reduce overhead
    let expect_ixj = cal_two_points_expects(detection_events);
    let mut expectations =
        Vec::from_iter(
            extended_hyperedges
                .iter()
                .take(n_low_order)
                .map(|e| match e.len() {
                    1 => expect_ixj[(e[0], e[0])],
                    2 => expect_ixj[(e[0], e[1])],
                    _ => {
                        unreachable!("First (n+1)n/2 elements must be 1- and 2-point expectations")
                    }
                }),
        );
    // calculate the rest of the expectations
    for hyperedge in extended_hyperedges.iter().skip(n_low_order) {
        expectations.push(
            hyperedge
                .iter()
                .fold(DVector::from_element(num_shots, 1.0_f64), |acc, &det| {
                    acc.component_mul(&detection_events.column(det))
                })
                .mean(),
        );
    }
    expectations
}

struct ClusterSolver {
    hyperedges: Vec<HyperEdge>,
    expectations: Vec<f64>,
    intersection: HashMap<HyperEdge, Vec<HyperEdge>>,
    supersets: HashMap<HyperEdge, Vec<Vec<HyperEdge>>>,
}

impl ClusterSolver {
    fn new(cluster: &Cluster, all_hyperedges: &[HyperEdge], expectations: &[f64]) -> Self {
        let hyperedges = cluster
            .iter()
            .map(|&i| all_hyperedges[i].clone())
            .collect_vec();
        let expectations = cluster.iter().map(|&i| expectations[i]).collect_vec();

        let mut intersection = HashMap::with_capacity(hyperedges.len());
        let mut supersets = HashMap::with_capacity(hyperedges.len());
        for hyperedge in &hyperedges {
            let intersect = intersects(hyperedge, &hyperedges);
            let superset = powerset(&intersect)
                .into_iter()
                .filter(|set| {
                    let sym_diff = symmetric_difference(set).collect_vec();
                    hyperedge.iter().all(|i| sym_diff.contains(i))
                })
                .collect_vec();
            intersection.insert(hyperedge.clone(), intersect);
            supersets.insert(hyperedge.clone(), superset);
        }
        Self {
            hyperedges,
            expectations,
            intersection,
            supersets,
        }
    }

    fn prob_within_cluster(
        &self,
        selected: &[HyperEdge],
        intersection: &[HyperEdge],
        probs: &[f64],
        filtering: Option<usize>,
    ) -> f64 {
        let mut prob = 1.0;
        self.hyperedges
            .iter()
            .enumerate()
            .filter(|&(_, h)| intersection.contains(h))
            .for_each(|(i, h)| {
                if let Some(filter) = filtering {
                    if i == filter {
                        return;
                    }
                }
                if selected.contains(h) {
                    prob *= probs[i];
                } else {
                    prob *= 1.0 - probs[i];
                }
            });
        prob
    }
}

impl Problem for ClusterSolver {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dim(&self) -> Self::Dim {
        Dynamic::new(self.hyperedges.len())
    }
}

impl System for ClusterSolver {
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        for (i, hyperedge) in self.hyperedges.iter().enumerate() {
            let mut local_cost = -self.expectations[i];
            let superset = &self.supersets[hyperedge];
            superset.iter().for_each(|select| {
                local_cost += self.prob_within_cluster(
                    select,
                    &self.intersection[hyperedge],
                    x.as_slice(),
                    None,
                )
            });
            fx[i] = local_cost;
        }
        Ok(())
    }
}

fn solve_cluster(
    cluster: &Cluster,
    all_hyperedges: &[HyperEdge],
    expectations: &[f64],
    atol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, Error> {
    let f = ClusterSolver::new(cluster, all_hyperedges, expectations);
    let dom = f.domain();
    let n_params = cluster.len();
    let mut solver = TrustRegion::new(&f, &dom);
    let mut x = DVector::from_element(n_params, 0.0);
    let mut fx = DVector::from_element(n_params, 0.0);

    for _ in 1..=max_iter {
        solver
            .next(&f, &dom, &mut x, &mut fx)
            .expect("solver encountered an error");
        if fx.norm() < atol {
            break;
        }
    }
    Ok(x.data.into())
}

fn adjust_probabilities(
    clusters: &[Cluster],
    all_hyperedges: &[HyperEdge],
    solved_probs: &[Vec<f64>],
) -> HashMap<HyperEdge, f64> {
    let mut adjusted_probs = HashMap::with_capacity(all_hyperedges.len());
    // cache for lengths
    let hyperedge_lengths: Vec<usize> = all_hyperedges.iter().map(|h| h.len()).collect();
    // Insert largest hyperedges directly without adjustment
    for (cluster, probs) in clusters.iter().zip(solved_probs) {
        let i = *cluster.last().unwrap();
        adjusted_probs.insert(all_hyperedges[i].clone(), *probs.last().unwrap());
    }

    let mut weight_to_adjust = all_hyperedges[*clusters[0].last().unwrap()].len() - 1;
    while weight_to_adjust > 0 {
        let mut collected_probs = HashMap::new();
        // adjust the probability of hyperedges with weight
        // weight_to_adjust in each clusters by the probability
        // of the hyperedges with weight greater than that
        for (cluster, probs) in clusters
            .iter()
            .zip(solved_probs)
            .filter(|&(cluster, _)| hyperedge_lengths[*cluster.last().unwrap()] > weight_to_adjust)
        {
            for (&hyperedge_i, &prob_this) in cluster
                .iter()
                .zip(probs)
                .filter(|&(i, _)| hyperedge_lengths[*i] == weight_to_adjust)
            {
                let hyperedge = &all_hyperedges[hyperedge_i];
                let adjusted_prob = adjusted_probs
                    .iter()
                    .filter_map(|(h, &p)| {
                        if cluster.iter().any(|&i| &all_hyperedges[i] == h)
                            || hyperedge.iter().any(|i| !h.contains(i))
                        {
                            return None;
                        }
                        Some(p)
                    })
                    .fold(prob_this, |p, q| (p - q) / (1.0 - 2.0 * q));
                collected_probs
                    .entry(hyperedge)
                    .or_insert_with(Vec::new)
                    .push(adjusted_prob);
            }
        }
        // average the probabilities of the same hyperedge in different clusters
        for (h, probs) in collected_probs {
            let prob = probs.iter().sum::<f64>() / probs.len() as f64;
            adjusted_probs.insert(h.clone(), prob);
        }
        weight_to_adjust -= 1;
    }
    adjusted_probs
}

#[inline]
fn powerset(hyperedges: &[HyperEdge]) -> Vec<Vec<HyperEdge>> {
    (1..=hyperedges.len())
        .flat_map(move |k| hyperedges.iter().cloned().combinations(k))
        .collect_vec()
}

#[inline]
fn intersects(target: &HyperEdge, others: &[HyperEdge]) -> Vec<HyperEdge> {
    others
        .iter()
        .filter(|&h| h.iter().any(|&e| target.contains(&e)))
        .cloned()
        .collect_vec()
}

#[inline]
fn symmetric_difference(hyperedges: &[HyperEdge]) -> impl Iterator<Item = usize> {
    let mut counts = HashMap::new();
    for hyperedge in hyperedges {
        for &e in hyperedge {
            *counts.entry(e).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .filter_map(|(e, c)| if c % 2 == 1 { Some(e) } else { None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_difference() {
        assert_eq!(
            symmetric_difference(&[
                HyperEdge::from_slice(&[0, 1, 2]),
                HyperEdge::from_slice(&[1, 2, 3]),
            ])
            .collect::<HashSet<usize>>(),
            HashSet::from([0, 3])
        );

        assert_eq!(
            symmetric_difference(&[
                HyperEdge::from_slice(&[0, 1, 2]),
                HyperEdge::from_slice(&[1, 2, 3]),
                HyperEdge::from_slice(&[2, 3, 4]),
            ])
            .collect::<HashSet<usize>>(),
            HashSet::from([0, 2, 4])
        )
    }
}
