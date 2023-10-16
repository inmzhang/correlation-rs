import stim

from correlation_py import cal_2nd_order_correlations, cal_high_order_correlations


def main():
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=11,
        rounds=11,
        after_reset_flip_probability=0.01,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = dem.compile_sampler()
    dets, _, _ = sampler.sample(shots=500000)

    res = cal_2nd_order_correlations(dets)
    bdy, edges = res.data
    print(bdy)
    print(edges)

    result = cal_high_order_correlations(dets)
    print(result)


if __name__ == "__main__":
    main()
