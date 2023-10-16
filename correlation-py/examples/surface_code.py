import stim

from correlation_py import (
    cal_2nd_order_correlations,
    cal_high_order_correlations,
    TannerGraph,
)


def main():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_reset_flip_probability=0.01,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = dem.compile_sampler()
    dets, _, _ = sampler.sample(shots=500000)
    graph = TannerGraph(dem)

    res = cal_2nd_order_correlations(dets)
    bdy, edges = res.data
    print(bdy)
    print(edges)

    result = cal_high_order_correlations(dets, graph.hyperedges)
    for k, v in graph.hyperedge_probs.items():
        print(f"Hyperedge: {k}, ideal: {v}, result: {result.get(k)}")


if __name__ == "__main__":
    main()
